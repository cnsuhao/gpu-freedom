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
  // for status column, all jobs go to this workflow
  // if you add, remove values, you need to change the function
  // JobQueueStatusToString
  JS_NEW                   = 0;
  JS_RETRIEVING_WORKUNIT   = 10;
  JS_WORKUNIT_RETRIEVED    = 20;
  JS_ACKNOWLEDGING         = 30;
  JS_READY                 = 40;
  JS_RUNNING               = 50;
  JS_COMPUTED              = 60;
  JS_TRANSMITTING_WORKUNIT = 70;
  JS_WORKUNIT_TRANSMITTED  = 80;
  JS_TRANSMITTING_RESULT   = 90;
  JS_COMPLETED             = 100;
  JS_WU_CLEANEDUP          = 900;
  JS_ERROR                 = 999;

type TJobStatus      = Longint;


type TDbJobQueueRow = record
   id              : Longint;
   jobdefinitionid : String;
   status          : Longint;
   statusdesc      : String;
   server_id       : Longint;
   jobqueueid      : String;
   job             : AnsiString;
   jobtype         : String;
   workunitjob,
   workunitjobpath,
   workunitresult,
   workunitresultpath : String;
   nodeid,
   nodename        : String;
   requireack,
   islocal         : Boolean;
   acknodeid,
   acknodename     : String;
   create_dt  : TDateTime;
   update_dt  : TDateTime;
   transmission_dt : TDateTime;
   transmissionid  : String;
   ack_dt          : TDateTime;
   reception_dt    : TDateTime;
end;

type TDbJobQueueTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    procedure insertOrUpdate(var row : TDbJobQueueRow);
    procedure delete(var row : TDbJobQueueRow);
    function  findRowInStatus(var row : TDbJobQueueRow; status : TJobStatus) : Boolean;
    function  findRowWithJobQueueId(var row : TDbJobQueueRow; jobqueueid : String) : Boolean;

  private
    procedure createDbTable();
    procedure fillRow(var row : TDbJobQueueRow);
  end;

function JobQueueStatusToString(status : Longint) : String;

implementation



constructor TDbJobQueueTable.Create(filename : String);
begin
  inherited Create(filename, 'tbjobqueue', 'id');
  createDbTable();
end;

function JobQueueStatusToString(status : Longint) : String;
begin
  Result := 'Unknown Satus in JobQueueStatusToString()';
  case status of
    JS_NEW                   : Result := 'NEW';
    JS_RETRIEVING_WORKUNIT   : Result := 'RETRIEVING_WORKUNIT';
    JS_WORKUNIT_RETRIEVED    : Result := 'WORKUNIT_RETRIEVED';
    JS_ACKNOWLEDGING         : Result := 'ACKNOWLEDGING';
    JS_READY                 : Result := 'READY';
    JS_RUNNING               : Result := 'RUNNING';
    JS_COMPUTED              : Result := 'COMPUTED';
    JS_TRANSMITTING_WORKUNIT : Result := 'TRANSMITING_WORKUNIT';
    JS_WORKUNIT_TRANSMITTED  : Result := 'WORKUNIT_TRANSMITTED';
    JS_TRANSMITTING_RESULT   : Result := 'TRANSMITTING_RESULT';
    JS_COMPLETED             : Result := 'COMPLETED';
    JS_WU_CLEANEDUP          : Result := 'WORKUNIT_CLEANED_UP';
    JS_ERROR                 : Result := 'ERROR';
  end;
end;

procedure TDbJobQueueTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('jobqueueid', ftString);
      FieldDefs.Add('jobdefinitionid', ftString);
      FieldDefs.Add('status', ftInteger);
      FieldDefs.Add('statusdesc', ftString);
      FieldDefs.Add('job', ftString);
      FieldDefs.Add('jobtype', ftString);
      FieldDefs.Add('workunitjob', ftString);
      FieldDefs.Add('workunitjobpath', ftString);
      FieldDefs.Add('workunitresult', ftString);
      FieldDefs.Add('workunitresultpath', ftString);
      FieldDefs.Add('nodeid', ftString);
      FieldDefs.Add('nodename', ftString);
      FieldDefs.Add('requireack', ftBoolean);
      FieldDefs.Add('islocal', ftBoolean);
      FieldDefs.Add('acknodeid', ftString);
      FieldDefs.Add('acknodename', ftString);
      FieldDefs.Add('create_dt', ftDateTime);
      FieldDefs.Add('update_dt', ftDateTime);
      FieldDefs.Add('transmission_dt', ftDateTime);
      FieldDefs.Add('transmissionid', ftString);
      FieldDefs.Add('ack_dt', ftDateTime);
      FieldDefs.Add('reception_dt', ftDateTime);
      FieldDefs.Add('server_id', ftInteger);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

procedure TDbJobQueueTable.insertOrUpdate(var row : TDbJobQueueRow);
var options : TLocateOptions;
begin
  options := [];
  if dataset_.Locate('jobqueueid', row.jobqueueid, options) then
      dataset_.Edit
  else
      dataset_.Append;

  dataset_.FieldByName('jobdefinitionid').AsString := row.jobdefinitionid;
  dataset_.FieldByName('status').AsInteger := row.status;
  dataset_.FieldByName('statusdesc').AsString := JobQueueStatusToString(row.status);
  dataset_.FieldByName('jobqueueid').AsString := row.jobqueueid;
  dataset_.FieldByName('job').AsString := row.job;
  dataset_.FieldByName('jobtype').AsString := row.jobtype;
  dataset_.FieldByName('workunitjob').AsString := row.workunitjob;
  dataset_.FieldByName('workunitjobpath').AsString := row.workunitjobpath;
  dataset_.FieldByName('workunitresult').AsString := row.workunitresult;
  dataset_.FieldByName('workunitresultpath').AsString := row.workunitresultpath;
  dataset_.FieldByName('nodeid').AsString := row.nodeid;
  dataset_.FieldByName('nodename').AsString := row.nodename;
  dataset_.FieldByName('requireack').AsBoolean := row.requireack;
  dataset_.FieldByName('islocal').AsBoolean := row.islocal;
  dataset_.FieldByName('acknodeid').AsString := row.acknodeid;
  dataset_.FieldByName('acknodename').AsString := row.acknodename;
  dataset_.FieldByName('create_dt').AsDateTime := row.create_dt;
  dataset_.FieldByName('update_dt').AsDateTime := row.update_dt;
  dataset_.FieldByName('transmission_dt').AsDateTime := row.transmission_dt;
  dataset_.FieldByName('transmissionid').AsString := row.transmissionid;
  dataset_.FieldByName('ack_dt').AsDateTime := row.ack_dt;
  dataset_.FieldByName('reception_dt').AsDateTime := row.reception_dt;
  dataset_.FieldByName('server_id').AsInteger := row.server_id;
  dataset_.Post;
  dataset_.ApplyUpdates;
  row.id := dataset_.FieldByName('id').AsInteger;
end;

procedure TDbJobQueueTable.delete(var row : TDbJobQueueRow);
var options : TLocateOptions;
begin
  options := [];
  if dataset_.Locate('jobdefinitionid', row.jobdefinitionid, options) then
    begin
      dataset_.Delete;
      dataset_.Post;
      dataset_.ApplyUpdates;
    end
  else
    begin
     raise Exception.Create('Internal error: Nothing to delete in tbjobqueue, jobdefinitionid was '+row.jobdefinitionid);
    end;
end;

function  TDbJobQueueTable.findRowInStatus(var row : TDbJobQueueRow; status : TJobStatus) : Boolean;
var options : TLocateOptions;
begin
 Result := false;
 options := [];
 if dataset_.Locate('status', status, options) then
   begin
     fillRow(row);
     Result := true;
   end;
end;

function  TDbJobQueueTable.findRowWithJobQueueId(var row : TDbJobQueueRow; jobqueueid : String) : Boolean;
var options : TLocateOptions;
begin
 Result := false;
 options := [];
 if dataset_.Locate('jobqueueid', jobqueueid, options) then
   begin
     fillRow(row);
     Result := true;
   end;
end;

procedure TDbJobQueueTable.fillRow(var row : TDbJobQueueRow);
begin
     row.jobdefinitionid := dataset_.FieldByName('jobdefinitionid').AsString;
     row.status          := dataset_.FieldByName('status').AsInteger;
     row.statusdesc      := dataset_.FieldByName('statusdesc').AsString;
     row.jobqueueid      := dataset_.FieldByName('jobqueueid').AsString;
     row.job             := dataset_.FieldByName('job').AsString;
     row.jobtype         := dataset_.FieldByName('jobtype').AsString;
     row.workunitjob     := dataset_.FieldByName('workunitjob').AsString;
     row.workunitjobpath := dataset_.FieldByName('workunitjobpath').AsString;
     row.workunitresultpath  := dataset_.FieldByName('workunitresultpath').AsString;
     row.nodeid          := dataset_.FieldByName('nodeid').AsString;
     row.nodename        := dataset_.FieldByName('nodename').AsString;
     row.requireack      := dataset_.FieldByName('requireack').AsBoolean;
     row.islocal         := dataset_.FieldByName('islocal').AsBoolean;
     row.acknodeid       := dataset_.FieldByName('acknodeid').AsString;
     row.acknodename     := dataset_.FieldByName('acknodename').AsString;
     row.create_dt       := dataset_.FieldByName('create_dt').AsDateTime;
     row.update_dt       := dataset_.FieldByName('update_dt').AsDateTime;
     row.transmission_dt := dataset_.FieldByName('transmission_dt').AsDateTime;
     row.transmissionid  := dataset_.FieldByName('transmissionid').AsString;
     row.ack_dt          := dataset_.FieldByName('ack_dt').AsDateTime;
     row.reception_dt    := dataset_.FieldByName('reception_dt').AsDateTime;
     row.server_id       := dataset_.FieldByName('server_id').AsInteger;
end;

end.
