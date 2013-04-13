unit workflowancestors;
{
  This class is the ancestor for all workflows.
  Tables which have a status column typically implement a child of
  TWorkflowAncestor.

  (c) 2013 HB9TVM and the Global Processing Unit Team

}
interface

uses SyncObjs, SysUtils,
     dbtablemanagers, jobqueuetables, jobqueuehistorytables, loggers;

type TWorkflowAncestor = class(TObject)
      constructor Create(var tableman : TDbTableManager; var logger : TLogger);
      destructor  Destroy;

    protected
      CS_       : TCriticalSection;
      logger_   : TLogger;
      tableman_ : TDbTableManager;
end;


type TJobQueueWorkflowAncestor = class(TWorkflowAncestor)
   protected
       function findRowInStatus(var row : TDbJobQueueRow; s : TJobStatus) : Boolean;
       function changeStatus(var row : TDbJobQueueRow; fromS, toS : TJobStatus; message : String) : Boolean;
end;

implementation

constructor TWorkflowAncestor.Create(var tableman : TDbTableManager; var logger : TLogger);
begin
  inherited Create;
  CS_ := TCriticalSection.Create;
  logger_ := logger;
  tableman_ := tableman;
end;

destructor TWorkflowAncestor.Destroy;
begin
  if Assigned(CS_) then CS_.Free;
  inherited Destroy;
end;

function TJobQueueWorkflowAncestor.changeStatus(var row : TDbJobQueueRow; fromS, toS : TJobStatus; message : String) : Boolean;
var
   dbqueuehistoryrow : TDbJobQueueHistoryRow;
begin
  Result := false;
  if row.status<>fromS then
        begin
          logger_.log(LVL_SEVERE, 'Internal error: Not possible to change status for jobqueue row '+row.jobdefinitionid+' from '+
                                   JobQueueStatusToString(fromS)+' because row has status '+JobQueueStatusToString(row.status));
          Exit;
        end;

  CS_.Enter;
  try
    row.status := toS;
    row.update_dt := Now;

    dbqueuehistoryrow.status     := row.status;
    dbqueuehistoryrow.jobqueueid := row.jobqueueid;
    dbqueuehistoryrow.message    := message;

    tableman_.getJobQueueTable().insertOrUpdate(row);
    tableman_.getJobQueueHistoryTable().insert(dbqueuehistoryrow);
    Result := true;
  except
    on e : Exception  do
        logger_.log(LVL_SEVERE, 'Internal error: Not possible to change status for jobqueue row '+row.jobdefinitionid+' from '+
                                   JobQueueStatusToString(fromS)+' because of Exception '+e.Message);
  end;

  CS_.Leave;
  if Result then logger_.log(LVL_DEBUG, 'Jobqueue row '+row.jobdefinitionid+' changed status from '+
                                         JobQueueStatusToString(fromS)+' to status '+JobQueueStatusToString(row.status));
end;

function TJobQueueWorkflowAncestor.findRowInStatus(var row : TDbJobQueueRow; s : TJobStatus) : Boolean;
begin
  Result := false;
  CS_.Enter;
  try
     Result := tableman_.getJobQueueTable().findRowInStatus(row, s);
  except
    on e : Exception  do
        logger_.log(LVL_SEVERE, 'Internal error: Not possible to retrieve row in status '+
                                   JobQueueStatusToString(s)+' because of Exception '+e.Message);
  end;

  CS_.Leave;
end;

end.
