CREATE TABLE tbchat (
   id int primary key,
   msg text,
   user varchar(32),
   create_dt text,
   channel varchar(32)
);

CREATE TABLE tbjobqueue (
   id int primary key,
   jobid text,
   job text,
   create_dt text
);

CREATE TABLE tbjobresult (
   id int primary key,
   jobid text,
   job text,
   result text,
   create_dt text
);

CREATE TABLE tbnode (
   id int primary key,   
   node text,
   speed int,
   ram int,
   mhz int,
   nbcpus int,
   isonline numeric,
   create_dt text
);

CREATE TABLE tbwanparameter (
   id int primary key,
   paramname text,
   paramgroup text,
   paramvalue text
);

CREATE TABLE tbparameter (
   id int primary key,
   paramname text,
   paramgroup text,
   paramvalue text
);


