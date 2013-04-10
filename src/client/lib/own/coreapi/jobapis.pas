unit jobapis;

interface

uses coreapis, dbtablemanagers, jobdefinitiontables, jobqueuetables, utils;


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
    constructor Create(var tableman : TDbTableManager);

    procedure createJob(var job : TGPUJobAPI);

end;


implementation

constructor TJobAPI.Create(var tableman : TDbTableManager);
begin
  inherited Create(tableman);
end;

procedure TJobAPI.createJob(var job : TGPUJobAPI);
var i : Longint;
    jobdefrow   : TDbJobDefinitionRow;
    jobqueuerow : TDbJobQueueRow;
begin
   jobdefrow.jobdefinitionid:=createUniqueId();

   for i:=1 to job.trandetails.nbrequests do
          begin
            jobqueuerow.jobqueueid := createUniqueId();
          end;

end;

end.
