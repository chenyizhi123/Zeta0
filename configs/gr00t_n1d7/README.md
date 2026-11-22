# GR00T N1.7 Zeta

[`joint.template.json`](joint.template.json) selects the native N1.7 host,
v6/raw objectives, horizon 40, four solver steps and explicit GR00T initialization.
Fill in the local paths and run a GPU preflight at your intended batch size.

Follow [the backend workflow](../../docs/backends.md) and
[native preparation](../../docs/gr00t_n1d7_preparation.md) to acquire pinned source,
export the backbone, verify base weight identity, prepare data and materialize a
new feature cache. Psi0 features/checkpoints cannot be substituted by changing a
path or backend flag. Native source and all model weights stay external.
