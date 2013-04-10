unit jobapis;

interface

uses SysUtils, coreapis, dbtablemanagers, servermanagers, jobdefinitiontables,
     jobqueuetables, jobqueuehistorytables, identities, loggers, utils, stkconstants;


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
    jobdefrow      : TDbJobDefinitionRow;
    jobqueuerow    : TDbJobQueueRow;
    jqhistoryrow   : TDbJobQueueHistoryRow;
    srv            : TServerRecord;
    srvid          : Longint;
    erroneous      : Boolean;
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
   logger_.log(LVL_INFO, logHeader_+'Jobdefinition '+jobdefrow.jobdefinitionid+' inserted into database.');

   for i:=1 to job.trandetails.nbrequests do
          begin
            erroneous := false;

            jobqueuerow.jobqueueid := createUniqueId();

            // this if statement selects either the client workflow or the
            // server workflow
            if job.islocal then
                jobqueuerow.status:=C_NEW
            else
                jobqueuerow.status:=S_NEW;

            jobqueuerow.jobdefinitionid:=jobdefrow.jobdefinitionid;
            jobqueuerow.islocal:=job.islocal;
            jobqueuerow.requireack:=job.requireack;
            jobqueuerow.job:=job.job;
            jobqueuerow.jobtype:=job.jobtype;

            jobqueuerow.workunitjob:=job.trandetails.workunitjob;
            if job.trandetails.tagwujob then
               jobqueuerow.workunitjob:=jobqueuerow.workunitjob+'_'+IntToStr(i);
            if jobqueuerow.workunitjob<>'' then
               jobqueuerow.workunitjobpath:=appPath_+WORKUNIT_FOLDER+PathDelim+INCOMING_WU_FOLDER+PathDelim+jobqueuerow.workunitjob;
            if not FileExists(jobqueuerow.workunitjobpath) then
                 begin
                   erroneous:= true;
                   logger_.log(LVL_ERROR, logHeader_+'Workunitjob '+jobqueuerow.workunitjobpath+' does not exist. Jobqueue will not be persisted!');
                 end;

            jobqueuerow.workunitresult:=job.trandetails.workunitresult;
            if job.trandetails.tagwuresult then
               jobqueuerow.workunitresult:=jobqueuerow.workunitresult+'_'+IntToStr(i);
            if jobqueuerow.workunitresult<>'' then
               jobqueuerow.workunitresultpath:=appPath_+WORKUNIT_FOLDER+PathDelim+OUTGOING_WU_FOLDER+PathDelim+jobqueuerow.workunitresult;

            jobqueuerow.serverstatus:='';
            jobqueuerow.jobresultid:='';
            jobqueuerow.nodeid:=myGPUID.NodeId;
            jobqueuerow.nodename:=myGPUID.Nodename;
            jobqueuerow.server_id:=srvid;
            jobqueuerow.create_dt:=Now;
            jobqueuerow.acknodeid:='';
            jobqueuerow.acknodename:='';
            jobqueuerow.ack_dt:=0;
            jobqueuerow.reception_dt:=0;

            if not erroneous then
               begin
                 tableman_.getJobQueueTable().insertOrUpdate(jobqueuerow);
                 // inserting also an entry into tbjobqueuehistory
                 jqhistoryrow.jobqueueid := jobqueuerow.jobqueueid;
                 jqhistoryrow.status := jobqueuerow.status;
                 if job.islocal then
                   jqhistoryrow.message := 'This is a local job created by TJobAPI'
                 else
                   jqhistoryrow.message := 'This is a global job created by TJobAPI';
                 jqhistoryrow.create_dt := Now;
                 tableman_.getJobQueueHistoryTable().insert(jqhistoryrow);

                 logger_.log(LVL_INFO, logHeader_+'Jobqueueid '+jobqueuerow.jobqueueid+' persisted :-)');
               end;
          end;

end;

end.
