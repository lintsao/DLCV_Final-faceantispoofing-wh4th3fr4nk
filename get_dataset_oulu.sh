# Download dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1251SwV6bnMrDF0EZ8kdmH2FZQTLSLbgU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1251SwV6bnMrDF0EZ8kdmH2FZQTLSLbgU" -O oulu.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./oulu.zip

# Remove the downloaded zip file
rm ./oulu.zip
