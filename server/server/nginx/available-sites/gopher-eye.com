server {
    listen 80;
    server_name gopher-eye.com;

    location / {
        proxy_pass http://localhost:5000;
    }   
}