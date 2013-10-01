library(RJDBC);
source("D:\\xampp\\htdocs\\gpu_freedom\\src\\playground\\R\\config.r")
drv <- JDBC("com.mysql.jdbc.Driver", "D:\\jdbc\\mysql-connector-java-5.1.5-bin.jar",identifier.quote="`");

b_conn  <- dbConnect(drv, "jdbc:mysql://127.0.0.1:3306/bitcoin", username, password);
b_pricetable <- dbReadTable(b_conn, "pricevalue")
b_price      <- dbGetQuery(b_conn, "select price from pricevalue order by id asc")
b_lastprice  <- dbGetQuery(b_conn, "select price from pricevalue where create_dt>(NOW() - INTERVAL 7 DAY)  order by id asc")


p_conn      <- dbConnect(drv, "jdbc:mysql://127.0.0.1:3306/powergrid", username, password);
p_freqtable <- dbReadTable(p_conn, "tbfrequency")
p_freq      <- dbGetQuery(p_conn, "select frequencyhz from tbfrequency order by id asc")
p_netdiff   <- dbGetQuery(p_conn, "select networkdiff from tbfrequency order by id asc")

f_conn <- dbConnect(drv, "jdbc:mysql://127.0.0.1:3306/finance", username, password);
f_tickerstable <- dbReadTable(f_conn, "tickers")
f_vix  <- dbGetQuery(f_conn, "select value from tickers where name='VIX' order by id asc")
f_sp   <- dbGetQuery(f_conn, "select value from tickers where name='GSPC' order by id asc")

# analyze(b_price$price)
# analyze(b_lastprice$price)
#
# analyze(p_freq$frequencyhz)
# analyze(p_netdiff$networkdiff)
#
# analyze(f_sp$value)
# analyze(f_vix$value)