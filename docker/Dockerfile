# 使用官方 Ruby 映像作為基礎
FROM ruby:3.3-alpine

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apk add --no-cache \
    build-base \
    git \
    nodejs \
    npm \
    tzdata

# 設定時區
ENV TZ=Asia/Taipei

# 複製 Gemfile、gemspec 和 package.json
COPY Gemfile* ./
COPY *.gemspec ./
COPY package*.json ./

# 安裝 Ruby 依賴
RUN bundle install

# 安裝 Node.js 依賴
RUN npm install

# 複製專案文件
COPY . .

# 暴露端口
EXPOSE 4000 35729

# 預設命令
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--livereload"]
