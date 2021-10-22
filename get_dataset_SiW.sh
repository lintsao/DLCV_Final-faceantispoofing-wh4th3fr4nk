# Download dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eUd3Y0_9y_xZ6CDDn3p9Y2KWKH997do_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eUd3Y0_9y_xZ6CDDn3p9Y2KWKH997do_" -O SiW.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./SiW.zip

# Remove the downloaded zip file
rm ./SiW.zip
