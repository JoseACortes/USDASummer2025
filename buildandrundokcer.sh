docker build -t my-ubuntu-workspace .
docker run -it -v $(pwd):/workspace my-ubuntu-workspace