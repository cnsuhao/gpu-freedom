PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE tbserver (
id AUTOINC_INT PRIMARY KEY , 
externalid INTEGER , 
servername VARCHAR(255) , serverurl VARCHAR(255) , chatchannel VARCHAR(255) , version VARCHAR(255) , online BOOL_INT , updated BOOL_INT , defaultsrv BOOL_INT , superserver BOOL_INT , uptime FLOAT , totaluptime FLOAT , longitude FLOAT , latitude FLOAT , distance FLOAT , activenodes INTEGER , jobinqueue INTEGER , failures INTEGER , create_dt DATETIME , update_dt DATETIME);
CREATE TABLE tbclient (
id AUTOINC_INT PRIMARY KEY , 
nodeid VARCHAR(255) , 
server_id INTEGER , 
nodename VARCHAR(255) , 
country VARCHAR(255) , 
region VARCHAR(255) , 
city VARCHAR(255) , 
zip VARCHAR(255) , 
description VARCHAR(255) , 
ip VARCHAR(255) , 
port VARCHAR(255) , 
localip VARCHAR(255) , 
os VARCHAR(255) , 
cputype VARCHAR(255) , 
version VARCHAR(255) , 
acceptincoming BOOL_INT , 
gigaflops INTEGER , 
ram INTEGER , 
mhz INTEGER , 
bits INTEGER , 
nbcpus INTEGER , 
online BOOL_INT , 
updated BOOL_INT , 
uptime FLOAT , 
totaluptime FLOAT , 
longitude FLOAT , 
latitude FLOAT , 
userid VARCHAR(255) , 
team VARCHAR(255) , 
create_dt DATETIME , 
update_dt DATETIME);
INSERT INTO "tbclient" VALUES(1,'4',0,'blabla',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'0',0,0,0,0,32,0,1,1,0.0,9.0,13.5,14.3,NULL,NULL,41338.7251161574,41338.7252205324);
INSERT INTO "tbclient" VALUES(2,'1',0,'andromeda','Switzerland',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'Win7',NULL,'0.5',0,0,0,0,0,0,1,1,0.0,0.0,7.0,46.5,NULL,NULL,41338.7251163426,41338.7252207176);
INSERT INTO "tbclient" VALUES(3,'2',0,'virgibuntu','Switzerland',NULL,NULL,NULL,NULL,NULL,NULL,NULL,'WinXP',NULL,'1.5',0,0,0,0,0,0,1,1,0.0,0.0,8.0,47.0,NULL,NULL,41338.7251165278,41338.7252209028);
CREATE TABLE tbchannel (id AUTOINC_INT PRIMARY KEY , externalid INTEGER , content VARCHAR(255) , user VARCHAR(255) , nodename VARCHAR(255) , nodeid VARCHAR(255) , channame VARCHAR(255) , chantype VARCHAR(255) , server_id INTEGER , create_dt DATETIME , usertime_dt DATETIME);
CREATE TABLE tbretrieved (id AUTOINC_INT PRIMARY KEY , lastmsg INTEGER , msgtype VARCHAR(255) , server_id INTEGER , create_dt DATETIME , update_dt DATETIME);
CREATE TABLE tbjob (id AUTOINC_INT PRIMARY KEY , externalid VARCHAR(255) , jobid VARCHAR(255) , job VARCHAR(255) , status INTEGER , workunitincoming VARCHAR(255) , workunitoutgoing VARCHAR(255) , requests INTEGER , delivered INTEGER , results INTEGER , islocal BOOL_INT , nodeid VARCHAR(255) , nodename VARCHAR(255) , server_id INTEGER , create_dt DATETIME);
CREATE TABLE tbjobresult (id AUTOINC_INT PRIMARY KEY , externalid INTEGER , requestid INTEGER , jobid VARCHAR(255) , job_id INTEGER , jobresult VARCHAR(255) , workunitresult VARCHAR(255) , iserroneous BOOL_INT , errorid INTEGER , errormsg VARCHAR(255) , errorarg VARCHAR(255) , server_id INTEGER , nodeid VARCHAR(255) , nodename VARCHAR(255) , create_dt DATETIME);
CREATE TABLE tbjobqueue (id AUTOINC_INT PRIMARY KEY , job_id INTEGER , requestid INTEGER , server_id INTEGER , create_dt DATETIME);
COMMIT;
