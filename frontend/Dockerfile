# Frontend static site with Nginx
FROM nginx:alpine

# Copy frontend files (index.html is in this directory)
COPY index.html /usr/share/nginx/html/index.html

# Custom nginx config for SPA
RUN echo 'server { \
    listen 8080; \
    root /usr/share/nginx/html; \
    index index.html; \
    location / { \
    try_files $uri $uri/ /index.html; \
    } \
    }' > /etc/nginx/conf.d/default.conf

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
