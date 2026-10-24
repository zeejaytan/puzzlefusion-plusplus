# 01: Build pytorch3d and chamferdist outside the repo; logs and renders out of it

**What to build:** The compiled dependencies live under the working area's `envs/` or `build/`, renders and logs go to run directories, and the repo checkout holds source only.

**Answers:** none (workbench routine).

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

Amendments 2.6 and 2.3. Rebuilding `pytorch3d` is slow: move it (and point the environment at the new path) rather than rebuild, and prove the environment still imports it on a compute node.

- [ ] `pytorch3d/` and `chamferdist/` no longer inside the checkout; imports still work
- [ ] No batch script writes logs or renders into the checkout
