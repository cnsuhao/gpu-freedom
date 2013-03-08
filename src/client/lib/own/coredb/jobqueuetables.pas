unit jobqueuetables;
{
   TDbJobTable contains the jobs which need to be executed or are already executed.
   If they need to be executed, there will be a reference in TDbJobQueue.
   If a result was computed, the result will be stored in TDbJobResult with a reference
    to the job.jobtables

   (c) by 2011 HB9TVM and the GPU Team
}
interface

uses sqlite3ds, db, coretables, SysUtils;

const
  // for status column
  JS_READY           = 30;
  JS_RUNNING         = 40;
  JS_ONHOLD          = 50;
  JS_COMPLETED_OK       = 90;
  JS_COMPLETED_ERROR    = 99;


type TDbJobQueueRow = record
   id              : Longint;
   jobdefinitionid : String;
   status          : Longint;
   server_id       : Longint;
   jobqueueid      : String;
   workunitjob,
   workunitresult  : String;
   nodeid,
   nodename        : String;
   requireack,
   islocal         : Boolean;
   acknodeid,
   acknodename     : String;
   create_dt  : TDateTime;
   transmission_dt : TDateTime;
   transmissionid  : String;
   ack_dt          : TDateTime;
   reception_dt    : TDateTime;
end;

type TDbJobQueueTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    procedure insert(var row : TDbJobQueueRow);
    procedure delete(var row : TDbJobQueueRow);

  private
    procedure createDbTable();
  end;

implementation

constructor TDbJobQueueTable.Create(filename : String);
begin
  inherited Create(filename, 'tbjobqueue', 'id');
  createDbTable();
end;

procedure TDbJobQueueTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('jobdefinitionid', ftString);
      FieldDefs.Add('status', ftInteger);
      FieldDefs.Add('server_id', ftInteger);
      FieldDefs.Add('jobqueueid', ftString);
      FieldDefs.Add('workunitjob', ftString);
      FieldDefs.Add('workunitresult', ftString);
      FieldDefs.Add('nodeid', ftString);
      FieldDefs.Add('nodename', ftString);
      FieldDefs.Add('requireack', ftBoolean);
      FieldDefs.Add('islocal', ftBoolean);
      FieldDefs.Add('acknodeid', ftString);
      FieldDefs.Add('acknodename', ftString);
      FieldDefs.Add('create_dt', ftDateTime);
      FieldDefs.Add('transmission_dt', ftDateTime);
      FieldDefs.Add('transmissionid', ftString);
      FieldDefs.Add('ack_dt', ftDateTime);
      FieldDefs.Add('reception_dt', ftDateTime);
      FieldDefs.Add('server_id', ftInteger);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

procedure TDbJobQueueTable.insert(var row : TDbJobQueueRow);
begin
  dataset_.Append;
  dataset_.FieldByName('job_id').AsInteger := row.job_id;
  dataset_.FieldByName('requestid').AsInteger := row.requestid;
  dataset_.FieldByName('server_id').AsInteger := row.server_id;
  dataset_.FieldByName('create_dt').AsDateTime := Now;
  dataset_.Post;
  dataset_.ApplyUpdates;
  row.id := dataset_.FieldByName('id').AsInteger;
end;

procedure TDbJobQueueTable.delete(var row : TDbJobQueueRow);
var options : TLocateOptions;
begin
  options := [];
  if dataset_.Locate('job_id', row.job_id, options) then
    begin
      dataset_.Delete;
      dataset_.Post;
      dataset_.ApplyUpdates;
    end
  else
    begin
     raise Exception.Create('Internal error: Nothing to delete in tbjobqueue, job_id was '+IntToStr(row.job_id));
    end;
end;

end.
