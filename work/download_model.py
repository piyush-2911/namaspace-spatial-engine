from pathlib import Path
import requests, sys
from kaggle import api
from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
slug=sys.argv[1] if len(sys.argv)>1 else 'namas-gaussian-splat-training'
out=Path(__file__).resolve().parent.parent/'outputs'
if slug=='namas-full-quality': out=out/'full-quality'
out.mkdir(exist_ok=True)
targets={'splat_output/meetingroom.ply','splat_output/checkpoint.pt'}
if slug=='namas-full-quality': targets.add('splat_output/capture_path_preview.mp4')
with api.build_kaggle_client() as client:
    token=None
    while True:
        req=ApiListKernelSessionOutputRequest()
        req.user_name=__import__('os').environ['KAGGLE_USERNAME']; req.kernel_slug=slug; req.page_size=200
        if token: req.page_token=token
        response=client.kernels.kernels_api_client.list_kernel_session_output(req)
        for item in response.files:
            if item.file_name in targets:
                dest=out/Path(item.file_name).name
                with requests.get(item.url,stream=True,timeout=(30,120)) as r:
                    r.raise_for_status()
                    with dest.with_suffix(dest.suffix+'.part').open('wb') as f:
                        for chunk in r.iter_content(1024*1024): f.write(chunk)
                dest.with_suffix(dest.suffix+'.part').replace(dest)
                print(dest.name,dest.stat().st_size,flush=True)
        token=response.next_page_token
        if not token: break
assert all((out/Path(p).name).exists() for p in targets)
