IMAGE="my-data-val-image"
AWS_ACCOUNT_ID=619071320705
REGION=ap-southeast-1
REMOTE_IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE
DIR=data-validate

# aws ecr update-repository --repository-name $IMAGE && \
docker build -t $IMAGE $DIR && \
docker tag $IMAGE:latest $REMOTE_IMAGE:latest && \
aws ecr get-login-password | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com && \
docker push $REMOTE_IMAGE:latest