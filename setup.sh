mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"aaronlorenzowoods@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=true\n\
enableXsrfProtection=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml