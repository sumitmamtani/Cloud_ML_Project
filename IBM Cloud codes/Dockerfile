FROM nginx:alpine

# Document who is responsible for this image
MAINTAINER Aditya Pandey "ap6624@nyu.edu"

WORKDIR /usr/share/nginx/html
COPY ui_ibm.html ./index.html

EXPOSE 80
EXPOSE 8000
EXPOSE 8080
EXPOSE 443

CMD ["nginx", "-g", "daemon off;"]
