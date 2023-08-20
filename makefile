start:
	docker build -t image-spngp .
	docker run --name container-spngp

init:
	docker exec -it container-spngp bash
	clear
	echo "Welcome to container"

results:
	python3 cccp-spngp.py
	python3 energy-spngp.py
	python3 concrete-spngp.py

