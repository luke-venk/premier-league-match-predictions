# Use when restarting the container after editing files, if you haven't changed Dockerfile or requirements.txt.
docker:
	docker compose down
	docker compose up

# Use if modified something that affected the image build, like Dockerfile or requirements.txt.
build:
	docker compose down
	docker compose build
	docker compose up

# Use if debugging weird dependency issues or you suspect Docker cache is corrupted, or you just want
# a completely fresh build.
clean:
	docker compose down
	docker compose build --no-cache
	docker compose up