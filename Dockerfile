# Sử dụng Nginx làm web server nhẹ nhất để phục vụ tĩnh SPA
FROM nginx:alpine

# Xóa cấu hình nginx mặc định
RUN rm -rf /usr/share/nginx/html/*

# Copy toàn bộ code Frontend (index.html, css, js, data) vào Nginx
COPY ./frontend /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Chạy Nginx
CMD ["nginx", "-g", "daemon off;"]
