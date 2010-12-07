CREATE TABLE tbchat (
   id int primary key,
   externalid int,
   msg text,
   user varchar(32),
   create_dt text,
   usertime_dt text,
   channel varchar(32),
   server_id int
);

CREATE TABLE tbjob (
   id int primary key,
   requestid int,
   jobid text,
   job text,
   status int,
   workunitincoming text,
   workunitoutgoing text,
   islocal boolean,
   server_id int,
   create_dt text
);

CREATE TABLE tbjobqueue (
   id int primary key,
   job_id int
);

CREATE TABLE tbjobresult (
   id int primary key,
   job_id int,
   result text,
   iserroneus boolean,
   reported boolean,
   errorid int,
   errormsg text,
   errorarg text,
   create_dt text
);

CREATE TABLE tbnode (
   id int primary key,
   defaultserver_id int,   
   nodeid text,
   nodename text,
   country text,
   region text,
   ip text,
   port int
   localip text,
   os text,
   version text,
   acceptincoming boolean,
   gigaflops int,
   ram int,
   mhz int,
   nbcpus int,
   online boolean,
   updated boolean,
   uptime real,
   totaluptime real,
   cputype text,
   longitude real,
   latitude real,
   create_dt text,
   update_dt text
);

CREATE TABLE tbserver (
   id int primary key,
   externalid int,
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



