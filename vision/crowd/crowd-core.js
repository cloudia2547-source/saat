// s'aat crowd navigation core: people tracking, Social Force prediction, and a wait / detour / back-off planner.
// Pure functions, no browser APIs, so the same file runs on the page and in tests.
//
// World frame: metres, x/y on the floor, angles in radians counter-clockwise.
// Robot: { x, y, th }  (th = heading the camera faces)
// Left of the robot is +angle, right is -angle.
(function (root) {
  "use strict";

  const DEF = {
    robotWidth: 0.40,     // m
    speed: 0.40,          // m/s
    personalSpace: 0.60,  // m, gap kept between robot edge and a person's body
    personR: 0.25,        // m, body radius
    fov: 64 * Math.PI / 180,
    horizon: 8.0,         // s of prediction (longest wait + lookahead)
    dt: 0.2,              // s per prediction step
    maxWait: 4.0,         // s, longest wait before taking a detour
    lookahead: 4.0,       // s of the robot's own path that must be clear after starting to move
  };

  const hyp = Math.hypot;
  const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

  // ---------------- tracking (alpha-beta filter per person) ----------------
  // tracks: [{ id, x, y, vx, vy, seen, hits, missed }]
  function updateTracks(tracks, meas, t, nextId) {
    const A = 0.55, B = 0.25, GATE = 0.9, MAXV = 2.5;
    const used = new Set();
    const out = [];
    // predict + greedy nearest association
    const pairs = [];
    for (const tr of tracks) {
      const dt = Math.max(0.01, t - tr.seen);
      const px = tr.x + tr.vx * dt, py = tr.y + tr.vy * dt;
      for (let j = 0; j < meas.length; j++) pairs.push({ tr, j, d: hyp(meas[j].x - px, meas[j].y - py), px, py, dt });
    }
    pairs.sort((a, b) => a.d - b.d);
    const done = new Set();
    for (const p of pairs) {
      if (p.d > GATE || done.has(p.tr) || used.has(p.j)) continue;
      done.add(p.tr); used.add(p.j);
      const m = meas[p.j], rx = m.x - p.px, ry = m.y - p.py;
      let vx = p.tr.vx + (B / p.dt) * rx, vy = p.tr.vy + (B / p.dt) * ry;
      const sp = hyp(vx, vy); if (sp > MAXV) { vx *= MAXV / sp; vy *= MAXV / sp; }
      out.push({ id: p.tr.id, x: p.px + A * rx, y: p.py + A * ry, vx, vy, seen: t, hits: p.tr.hits + 1, missed: 0 });
    }
    // keep briefly-unseen tracks (coasting), drop old ones
    for (const tr of tracks) {
      if (done.has(tr)) continue;
      const age = t - tr.seen;
      if (age < 1.2) out.push({ ...tr, missed: tr.missed + 1 });
    }
    // new tracks
    for (let j = 0; j < meas.length; j++) {
      if (used.has(j)) continue;
      out.push({ id: nextId.v++, x: meas[j].x, y: meas[j].y, vx: 0, vy: 0, seen: t, hits: 1, missed: 0 });
    }
    return out;
  }

  // position of a track at time t (coasting tracks extrapolated a little)
  function trackNow(tr, t) {
    const dt = Math.min(0.6, Math.max(0, t - tr.seen));
    return { x: tr.x + tr.vx * dt, y: tr.y + tr.vy * dt };
  }

  // ---------------- groups: people standing together ----------------
  function standingGroups(people, p) {
    const still = people.filter(q => hyp(q.vx, q.vy) < 0.25);
    const obs = [];
    for (let i = 0; i < still.length; i++) for (let j = i + 1; j < still.length; j++) {
      const a = still[i], b = still[j], d = hyp(a.x - b.x, a.y - b.y);
      if (d < 1.5) obs.push({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, r: Math.max(p.personR, d / 2) });
    }
    return obs;
  }

  // ---------------- Social Force prediction (Helbing & Molnar 1995) ----------------
  // people: [{ id, x, y, vx, vy }] at time 0. Returns [{ id, still, pts: [{x, y}] }] one point per dt.
  function predict(people, robot, params) {
    const p = { ...DEF, ...params };
    const steps = Math.round(p.horizon / p.dt);
    const TAU = 0.5, A = 2.1, Bf = 0.3, LAMBDA = 0.35;
    const robotR = p.robotWidth / 2;
    const st = people.map(q => {
      const sp = hyp(q.vx, q.vy);
      return { id: q.id, x: q.x, y: q.y, vx: q.vx, vy: q.vy, v0: sp > 0.25 ? Math.min(sp, 1.6) : 0,
               ex: sp > 0.25 ? q.vx / sp : 0, ey: sp > 0.25 ? q.vy / sp : 0, pts: [{ x: q.x, y: q.y }] };
    });
    const rep = (s, ox, oy, rsum, out) => {
      const dx = s.x - ox, dy = s.y - oy, d = Math.max(0.05, hyp(dx, dy));
      const nx = dx / d, ny = dy / d;
      const cosPhi = -(nx * s.ex + ny * s.ey);                 // obstacle ahead of the walker -> stronger
      const w = LAMBDA + (1 - LAMBDA) * (1 + cosPhi) / 2;
      const mag = A * Math.exp((rsum - d) / Bf) * w;
      out.fx += mag * nx; out.fy += mag * ny;
    };
    for (let k = 0; k < steps; k++) {
      const F = st.map(s => {
        if (s.v0 === 0) return { fx: 0, fy: 0 };                 // standing people stay put
        const f = { fx: (s.v0 * s.ex - s.vx) / TAU, fy: (s.v0 * s.ey - s.vy) / TAU };
        for (const o of st) if (o !== s) rep(s, o.x, o.y, 2 * p.personR, f);
        rep(s, robot.x, robot.y, p.personR + robotR, f);          // people step around s'aat
        return f;
      });
      st.forEach((s, i) => {
        if (s.v0 === 0) { s.pts.push({ x: s.x, y: s.y }); return; }
        s.vx += F[i].fx * p.dt; s.vy += F[i].fy * p.dt;
        const sp = hyp(s.vx, s.vy), cap = 1.3 * s.v0;
        if (sp > cap) { s.vx *= cap / sp; s.vy *= cap / sp; }
        s.x += s.vx * p.dt; s.y += s.vy * p.dt;
        s.pts.push({ x: s.x, y: s.y });
      });
    }
    return st.map(s => ({ id: s.id, still: s.v0 === 0, pts: s.pts }));
  }

  // ---------------- planner ----------------
  function at(pred, t, dt) { const i = Math.min(pred.pts.length - 1, Math.max(0, Math.round(t / dt))); return pred.pts[i]; }

  // first conflict along a robot path; returns ids of people it would come too close to
  function conflicts(path, preds, obs, p) {
    const need = p.robotWidth / 2 + p.personR + p.personalSpace;
    const ids = new Set();
    for (const q of path) {
      for (const pr of preds) { const o = at(pr, q.t, p.dt); if (hyp(o.x - q.x, o.y - q.y) < need) ids.add(pr.id); }
      for (const g of obs) if (hyp(g.x - q.x, g.y - q.y) < p.robotWidth / 2 + g.r + p.personalSpace * 0.5) ids.add("group");
    }
    return ids;
  }

  // robot route: wait `delay` s in place, then drive through `wps` (list of {x, y}).
  // Sampled every dt until the prediction horizon (beyond that nothing is known) or the end.
  function routeFor(robot, wps, delay, p) {
    const pts = [], until = Math.min(p.horizon, delay + Math.max(p.lookahead, 4.0));
    for (let t = 0; t <= delay + 1e-9 && t <= p.horizon; t += p.dt) pts.push({ t, x: robot.x, y: robot.y });
    let cx = robot.x, cy = robot.y, t = delay, length = 0;
    const legs = [];
    for (const w of wps) {
      const L = hyp(w.x - cx, w.y - cy); legs.push(Math.atan2(w.y - cy, w.x - cx)); length += L;
      const T = L / p.speed;
      for (let s = p.dt; s <= T + 1e-9; s += p.dt) {
        if (t + s > until) break;
        pts.push({ t: t + s, x: cx + (w.x - cx) * s / T, y: cy + (w.y - cy) * s / T });
      }
      t += T; cx = w.x; cy = w.y;
    }
    return { pts, length, legs, wps };
  }

  // decide the next move. goal: {x, y} or null (patrol straight ahead). prev: previous decision for smoothing.
  function plan(robot, goal, people, params, prev) {
    const p = { ...DEF, ...params };
    const target = goal || { x: robot.x + 4 * Math.cos(robot.th), y: robot.y + 4 * Math.sin(robot.th) };
    const preds = predict(people, robot, p);
    const obs = standingGroups(people, p);
    const goalRel = wrap(Math.atan2(target.y - robot.y, target.x - robot.x) - robot.th);
    const goalDist = hyp(target.x - robot.x, target.y - robot.y);
    const robotR = p.robotWidth / 2;

    // 1) back off: someone close and walking at s'aat
    for (const q of people) {
      const dx = robot.x - q.x, dy = robot.y - q.y, d = hyp(dx, dy);
      const closing = (dx * q.vx + dy * q.vy) / Math.max(d, 0.05);
      if (d < 1.0 + robotR && closing > 0.3) {
        return { action: "BACK", rel: 0, delay: 0, reason: "Person walking toward s'aat", ids: [q.id], preds, obs, path: null, cost: 0 };
      }
    }

    // 2) score options: straight to the goal now, wait then straight, or a detour through a side point
    const opts = [];
    const add = (wps, delay, detour) => {
      const r = routeFor(robot, wps, delay, p);
      const ids = conflicts(r.pts, preds, obs, p);
      const rel = wrap(r.legs[0] - robot.th);
      let turn = Math.abs(rel);
      for (let i = 1; i < r.legs.length; i++) turn += Math.abs(wrap(r.legs[i] - r.legs[i - 1]));
      const unseen = Math.abs(rel) > p.fov / 2 - 0.09 ? 1.5 : 0;                  // people out there are not visible
      opts.push({ rel, delay, detour, ids, ok: ids.size === 0, path: r,
                  cost: delay + r.length / p.speed + turn / (Math.PI / 2) * 1.0 + unseen });   // ~1 s per 90 degrees of turning
    };
    add([target], 0, false);
    for (let w = 0.5; w <= p.maxWait + 1e-9; w += 0.5) add([target], w, false);
    // side points laid out along the line to the goal: forward distance x sideways offset
    const gb = robot.th + goalRel, fx = Math.cos(gb), fy = Math.sin(gb);
    for (const f of [0.6, 1.2, 1.8, 2.4, 3.0]) {
      if (f > goalDist + 0.6) continue;
      for (const s of [-2.0, -1.5, -1.0, -0.6, 0.6, 1.0, 1.5, 2.0]) {
        const wp = { x: robot.x + fx * f - fy * s, y: robot.y + fy * f + fx * s };
        add([wp, target], 0, true);
      }
    }

    const direct = opts[0];
    const valid = opts.filter(o => o.ok).sort((a, b) => a.cost - b.cost);
    if (!valid.length) {
      return { action: "WAIT", rel: goalRel, delay: 0, blocked: true, reason: `Blocked by ${countPeople(direct.ids)}`,
               ids: [...direct.ids], preds, obs, path: direct.path, cost: Infinity };
    }
    let best = valid[0];
    // keep the previous choice if it is still safe and nearly as good (stops flickering between options)
    if (prev && prev.action !== "BACK" && !prev.blocked) {
      const same = valid.find(o => (o.delay === 0) === (prev.delay === 0) && o.detour === prev.detour && Math.abs(wrap(o.rel - prev.rel)) < 0.2);
      if (same && same.cost <= best.cost * 1.15 + 0.3) best = same;
    }
    let action, reason;
    if (best.delay > 0) {
      action = "WAIT";
      reason = `Waiting ~${best.delay.toFixed(1)} s for ${countPeople(direct.ids)} to pass`;
    } else {
      action = Math.abs(best.rel) <= 12 * Math.PI / 180 ? "GO" : best.rel > 0 ? "LEFT" : "RIGHT";
      if (!best.detour) reason = goal ? (goalDist < 0.5 ? "Almost there" : "Path clear") : "Path clear, looking for trash";
      else reason = `Going around ${countPeople(direct.ids)}`;
    }
    return { action, rel: best.rel, delay: best.delay, blocked: false, reason, ids: [...direct.ids], preds, obs, path: best.path, cost: best.cost };
  }

  function countPeople(ids) {
    const n = [...ids].filter(i => i !== "group").length, g = ids.has ? ids.has("group") : [...ids].includes("group");
    if (n === 0 && g) return "a group of people";
    const s = n === 1 ? "1 person" : `${n} people`;
    return g ? `${s} and a group` : s;
  }

  root.SaatCrowd = { DEF, updateTracks, trackNow, predict, plan, standingGroups, wrap };
})(typeof globalThis !== "undefined" ? globalThis : this);
