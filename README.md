# KataGo on Browser
KataGo powered by WebAssembly

## Install
### Convert a weight for TensorFlow.js

```sh
# NOTE: The specifying branch is IMPORTANT.
# NOTE: The below works on Apple Sicilon Mac. It does not work on Intel Mac.

git clone -b browser_v1.8.0 https://github.com/y-ich/KataGo.git # This branch support TensorFlow KataGo weights.
cd KataGo
mkdir models
cd models
curl -OL https://media.katagotraining.org/uploaded/networks/zips/kata1/kata1-b6c96-s175395328-d26788732.zip
unzip kata1-b6c96-s175395328-d26788732
cd ../tfjs
pipenv install
pipenv shell
make
cd ..
```

### Build

```sh
git checkout browser
source em_build.sh
```

### Start a web server

```sh
cd web
statikk --coi # or your favorite one liner server
```

### Open Browser
for auto detection of backend,
```
http://127.0.0.1:8080/?config=gtp_auto.cfg&model=web_model
```

Enjoy!
-
by ICHIKAWA, Yuji
