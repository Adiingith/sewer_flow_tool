FROM node:20

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# CRA can read the variables
COPY .env .env

ENV HOST 0.0.0.0

CMD ["npm", "start"]


# # build production version
# RUN npm run build || (echo "Build failed" && exit 1)

# # install serve as static server
# RUN npm install -g serve

# # start production server
# CMD ["serve", "-s", "build", "-l", "3000"]
