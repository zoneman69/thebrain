.PHONY: train_cloud release docker-image

train_cloud:
	python scripts/train_cloud.py --quick --output-dir artifacts

release: train_cloud
	python scripts/pull_push_artifacts.py upload --repo $${REPO} --file artifacts/manifest.json --tag $${TAG} --name manifest.json --content-type application/json --overwrite
	for file in window_fusion.pt fusion_gate.pt mix.pt time_mixer.pt; do \
	  python scripts/pull_push_artifacts.py upload --repo $${REPO} --file artifacts/$$file --tag $${TAG} --overwrite; \
	done
	@dir=$$(pwd)/artifacts/decoders; \
	for file in $$dir/*.pt; do \
	  python scripts/pull_push_artifacts.py upload --repo $${REPO} --file $$file --tag $${TAG} --overwrite; \
	done


docker-image:
	docker build -t hippocampus-train:latest .
