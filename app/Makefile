version := 1
project_id := jjkoh95
service_name := grab

docker-build:
	docker build -t ${service_name}:${version} .
docker-az-tag:
	docker tag ${service_name}:${version} vanagrab.azurecr.io/${service_name}:${version}
az-push-container-registry:
	docker push vanagrab.azurecr.io/${service_name}:${version}
az-deploy-container-instance:
	az container create --resource-group grab --name vanagrab${version} \
	--image vanagrab.azurecr.io/${service_name}:${version} \
	--location southeastasia \
	--dns-name-label vanagrab${version} --ports 80
az-build-deploy:
	make docker-build
	make docker-az-tag
	make az-push-container-registry
	make az-deploy-container-instance