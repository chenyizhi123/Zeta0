"""Serve the existing online protocol with the verified local model loader."""
import argparse
from http.server import ThreadingHTTPServer
import json
import logging
from pathlib import Path

import torch

from local_policy_runtime import LocalPolicy


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, help='Historical task or custom task declared by --model-paths')
    parser.add_argument('--model-paths', type=Path, default=None)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8014)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    torch.set_num_threads(4)
    from policy_profile import model_paths
    paths = model_paths(args.task, args.model_paths)
    loaded = LocalPolicy(paths['tasks'][args.task]['weights_dir'],paths['qwen_model'],
                         ROOT/'scripts/zeta0_network.py',args.device,vlm_source=paths.get('vlm_source'))
    # Preserve the delivered server's health/reset/infer protocol, session and
    # RTC validation. Only model loading uses the tested local CUDA migration.
    from psi.deploy.real_sonic_server import InferenceService, make_handler
    from runtime_audit import RuntimeAudit,audit_path,request_fields
    runtime = loaded.model.runtime
    manifest=runtime.read_json('inference_manifest.json')
    if args.task == 'tidy_scene' and args.device.startswith('cuda'):
        # Warm up both RTC shapes before advertising health. The first CUDA
        # inference can exceed this model's 0.5s prefix budget.
        from tidy_scene_offline import DATA, requests
        from psi.sonic_actions import controller_command_values
        import numpy as np
        request = next(requests(manifest))
        with np.load(DATA/'samples.npz',allow_pickle=False) as source:
            prefix = controller_command_values(source['actions'][0,:15]).tolist()
        for committed in ([], prefix):
            warmup = dict(request,committed_prefix=committed)
            prediction = runtime.infer_request(loaded.model.policy,loaded.model.processor,warmup,seed=args.seed)
            loaded.adapter.action_chunk(controller_command_values(prediction))
        logging.info('tidy_scene CUDA warmup complete (RTC prefixes 0 and 15); no robot commands sent')
    with RuntimeAudit(audit_path(ROOT,'server'),dict(task=args.task,
            weights_sha256=manifest['weights_sha256'],
            rtc_capacity_steps=runtime.rtc_max_prefix_steps,qwen_model=str(paths['qwen_model']))) as audit:
        class AuditedService(InferenceService):
            def health(self):
                verification = loaded.identity.get('provenance_verification', {})
                return dict(super().health(),rtc_audit_enabled=True,rtc_audit_log=str(audit.path),
                    weights_dir=str(loaded.adapter.weights_dir),
                    source_weights_sha256=manifest.get('source_weights_sha256'),
                    source_code_revision=manifest.get('source_code_revision'),
                    vlm_source=paths.get('vlm_source', 'packaged_qwen_training_base'),
                    qwen_model=loaded.identity['qwen_model'],
                    encoder_id=verification.get('encoder_id'),
                    matches_packaged_encoder=verification.get('encoder_id') == manifest.get('encoder_id'))

        def infer(request,seed):
            audit.record('server_request',**request_fields(request),seed=seed)
            try:
                actions=runtime.infer_request(loaded.model.policy,loaded.model.processor,request,seed=seed)
            except Exception as error:
                audit.record('server_error',session_id=request['session_id'],request_seq=request['request_seq'],
                             reason=repr(error))
                raise
            audit.record('server_prediction',session_id=request['session_id'],request_seq=request['request_seq'],
                         rtc_prefix_steps=len(request.get('committed_prefix',[])),actions=actions.copy())
            return actions
        service = AuditedService(infer,manifest, seed=args.seed,
            rtc_max_prefix_steps=runtime.rtc_max_prefix_steps,observation_contract=runtime.observation_contract)
        server = ThreadingHTTPServer((args.host,args.port),make_handler(service))
        server.daemon_threads = True
        logging.info('RTC diagnostics: %s',audit.path)
        logging.info('Online model ready: task=%s http://%s:%d; model service does not publish robot commands',
                     args.task,args.host,args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
