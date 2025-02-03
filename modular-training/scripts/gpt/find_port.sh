find_free_port() {
  while true; do
    PORT=$(shuf -i 6000-8000 -n 1)  # 随机选择一个6000-8000之间的端口
    ss -tuln | grep ":$PORT " > /dev/null  # 检查端口是否被占用
    if [ $? -ne 0 ]; then
      echo $PORT
      return
    fi
  done
}

echo $(find_free_port)