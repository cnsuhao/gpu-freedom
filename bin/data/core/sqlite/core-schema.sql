CREATE TABLE tbchannel (
   id int primary key,
   externalid int,
   content text,
   user varchar(32),
   nodename varchar(32),
   nodeid text,
   create_dt text,
   usertime_dt text,
   channame varchar(32),
   chantype varchar(32),
   server_id int
);

CREATE TABLE tbretrieved (
   id int primary key,
   lastmsg int,
   msgtype text,
   server_id int,
   create_dt text,
   update_dt text
)


CREATE TABLE tbjob (
   id int primary key,
   externalid int,
   jobid text,
   job text,
   status int,
   workunitincoming text,
   workunitoutgoing text,
   islocal boolean,
   nodeid text,
   nodename text,
   server_id int,
   create_dt text
);

CREATE TABLE tbjobqueue (
   id int primary key,
   job_id int,
   requestid int,
   server_id int,
   create_dt text
);

CREATE TABLE tbjobresult (
   id int primary key,
   externalid int,
   requestid int,
   jobid text,
   job_id int,
   jobresult text,
   workunitresult text,
   iserroneus boolean,
   errorid int,
   errormsg text,
   errorarg text,
   server_id int,
   nodename text,
   nodeid text,
   create_dt text
);

CREATE TABLE tbclient (
   id int primary key,
   nodeid text,
   nodename text,
   server_id int,
   country text,
   region text,
   city text,
   zip text,
   description text,
   ip text,
   port text,
   localip text,
   os text,
   version text,
   acceptincoming boolean,
   gigaflops int,
   ram int,
   mhz int,
   nbcpus int,
   bits int,
   isscreensaver boolean,
   online boolean,
   updated boolean,
   uptime real,
   totaluptime real,
   cputype text,
   userid text,
   longitude real,
   latitude real,
   description text,
   create_dt text,
   update_dt text
);

CREATE TABLE tbserver (
   id int primary key,
   externalid int,
   servername text,
   serverurl text,
   chatchannel varchar(32),
   version varchar(32),
   online boolean,
   updated boolean,
   defaultsrv boolean,
   superserver boolean,
   lon real,
   lat real,
   distance real,
   uptime real,
   totaluptime real,
   activenodes int,
   failures int,
   jobinqueue int
);

CREATE TABLE tbwanparameter (
   id int primary key,
   paramname text,
   paramgroup text,
   paramvalue text,
   create_dt text,
   update_dt text
);

CREATE TABLE tbparameter (
   id int primary key,
   paramname text,
   paramgroup text,
   paramvalue text,
   create_dt text,
   update_dt text
);

CREATE TABLE tbsynchronize
(
  projectname text,
  update_dt text,
  update_user text,
  update_type text,
  versionnr int primary key,
  branchname text,
  description text,
  update_fromversion int,
  update_fromsource text,
  schemaname text,
  dbtype text
);


INSERT INTO tbsynchronize (PROJECTNAME, VERSIONNR, BRANCHNAME, UPDATE_USER, UPDATE_TYPE, SCHEMANAME, DBTYPE)
VALUES ('deltasql-Server', 1, 'HEAD', 'INTERNAL', 'deltasql-server', '', 'sqlite');



