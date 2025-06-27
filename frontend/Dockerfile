FROM node:20

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# 让 CRA 正确读取变量
COPY .env .env

ENV HOST 0.0.0.0

CMD ["npm", "start"]


# # 构建生产版本
# RUN npm run build || (echo "Build failed" && exit 1)

# # 安装 serve 作为静态服务器
# RUN npm install -g serve

# # 启动生产服务器
# CMD ["serve", "-s", "build", "-l", "3000"]
