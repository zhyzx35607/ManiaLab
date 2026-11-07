import sys
from huggingface_hub import snapshot_download

snapshot_download("OliBomby/Mapperatorinator-v32", revision=sys.argv[2], local_dir=sys.argv[1],
                  allow_patterns=["gamemode=3/*", "README.md"])
