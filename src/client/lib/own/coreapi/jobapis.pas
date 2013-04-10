unit jobapis;

interface

uses SysUtils, coreapis, dbtablemanagers, servermanagers, jobdefinitiontables,
     jobqueuetables, identities, loggers, utils;


type TJobTransmissionDetails = record
     workunitjob : String;
     workunitresult : String;
     nbrequests  : Longint;
     tagwujob,
     tagwuresult : Boolean;
end;

TGPUJobAPI = record
    jobdefinitionid : String; // will be filled after call of createJob
    job             : AnsiString;
    jobtype         : String;
    requireack,
    islocal         : Boolean;

    trandetails     : TJobTransmissionDetails;
end;


type TJobAPI = class(TCoreAPI)
  public
    constructor Create(var tableman : TDbTableManager; var servman : TServerManager;
                       var logger : TLogger);

    // calling createJob will define a unique id for jobdefinitionid
    procedure createJob(var job : TGPUJobAPI);

end;


implementation

constructor TJobAPI.Create(var tableman : TDbTableManager;
                           var servman : TServerManager; var logger : TLogger);
begin
  inherited Create(tableman, servman, logger);
   logHeader_ := 'TJobAPI';
end;

procedure TJobAPI.createJob(var job : TGPUJobAPI);
var i : Longint;
    jobdefrow   : TDbJobDefinitionRow;
    jobqueuerow : TDbJobQueueRow;
    srv         : TServerRecord;
    srvid       : Longint;
begin
   servMan_.getDefaultServer(srv);

  if job.islocal then
      srvid := -1
   else
      srvid := srv.id;

   jobdefrow.jobdefinitionid:=createUniqueId();
   jobdefrow.job:=job.job;
   jobdefrow.jobtype:=job.jobtype;
   jobdefrow.requireack:=job.requireack;
   jobdefrow.islocal:=job.islocal;
   jobdefrow.nodeid:=myGPUID.NodeId;
   jobdefrow.nodename:=myGPUID.Nodename;
   jobdefrow.server_id:=srvid;

   jobdefrow.create_dt:=Now;
   jobdefrow.update_dt:=Now;

   tableman_.getJobDefinitionTable().insertOrUpdate(jobdefrow);

   for i:=1 to job.trandetails.nbrequests do
          begin
            jobqueuerow.jobqueueid := createUniqueId();
            //jobqueuerow.
          end;

end;

end.
