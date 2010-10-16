CREATE TABLE tbchat (
   id int primary key,
   msg text,
   user varchar(32),
   create_dt text,
   usertime_dt text,
   channel varchar(32)
);

CREATE TABLE tbjobqueue (
   id int primary key,
   jobid text,
   job text,
   workunitincoming text,
   workunitoutgoing text,
   create_dt text
);

CREATE TABLE tbjobresult (
   id int primary key,
   jobid text,
   job text,
   result text,
   workunit_incoming text,
   workunit_outgoing text,
   iserroneus numeric,
   errorid int,
   errormsg text,
   errorarg text,
   create_dt text
);

CREATE TABLE tbnode (
   id int primary key,   
   nodeid text,
   nodename text,
   country text,
   region text,
   ip text,
   port int
   localip text,
   os text,
   version text,
   acceptincoming numeric,
   gigaflops int,
   ram int,
   mhz int,
   nbcpus int,
   isonline numeric,
   uptime real,
   totaluptime real,
   cputype text,
   longitude real,
   latitude real,
   create_dt text,
   update_dt text
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


