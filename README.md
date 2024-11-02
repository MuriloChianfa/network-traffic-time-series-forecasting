# network-traffic-time-series-forecasting
Predict the future of your network with time series ML models.

# Getting started

Preprocessing raw netflow data

```bash
cd src/preprocess
go mod init network-traffic-time-series-forecasting
go mod tidy
go env -w GO111MODULE=on
go get github.com/phaag/go-nfdump@d2ff6042cb5186ede4064cbd50253ab97a78a89e
go run extract-traffic.go
```

Running model evaluations

```bash
cd src/forecast
virtualenv -p python3.11 venv
. ./venv/bin/activate
pip install -r requirements.txt
```
