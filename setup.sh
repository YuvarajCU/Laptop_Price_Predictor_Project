mkdir -p ~/.streamlit/
echo "
[server]n
port = $PORTn
enableCORS = falsen
headless = truen
n
" > ~/.streamlit/config.toml