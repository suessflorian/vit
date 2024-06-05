# Spinning up a VisionTransformer (ViT)
This hopefully serves to onramp quickly with a solid toolchain for training models (even beyond ViT's).

```sh
pipenv install
pipenv run python everything.py
```

Then start getting familiar with `mps`/`cuda` laying your tensor work via

```sh
pipenv run python everything.py --device mps
```

---
Notice how the ephemeral instance running setup (look at `./infra/main.tf`) is simply a recurring model "checkpoint" push to a GCS bucket. This mode can be set via `--gcs`... You will need to run some sort of bootstrap script (for me I've got `./infra/bootstrap.sh` that installs globally wrt. to the instance via `pip` all dependencies).

To interact with your spun up instance play with native `gcloud` instance administration tools like the amazing, loading up your instance with files here (like an ape, but it gets the job done).

```sh
gcloud compute scp *.py Pipfile Pipfile.lock <your-username>@<instance-name>:~/.
```

```sh
gcloud compute ssh "<instance-name>" --project "<project-name>"
```

_Note: you might have to get familiar with the image used, I have yet to align perfect the versions of `torch`/`torchvision` locally to what is actually available up there_
