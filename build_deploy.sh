export GCLOUD_PROJECT="dungs-poc"
export REPO="dungs-poc"
export REGION="europe-west1"
export IMAGE="dungs-poc"
export IMAGE_TAG=${REGION}-docker.pkg.dev/$GCLOUD_PROJECT/$REPO/$IMAGE

# next line only needed once
#gcloud auth configure-docker ${REGION}-docker.pkg.dev

docker build --platform linux/amd64 -t $IMAGE_TAG --no-cache \
.
docker push $IMAGE_TAG
