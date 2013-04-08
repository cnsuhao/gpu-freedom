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

  // stati for processing on client, prefix C_ for client
  // these stati are when we process a job for someone else
  C_NEW                     = 100;
  C_FOR_WU_RETRIEVAL        = 110;
  C_RETRIEVING_WORKUNIT     = 115;
  C_WORKUNIT_RETRIEVED      = 120;
  C_FOR_ACKNOWLEDGEMENT     = 130;
  C_ACKNOWLEDGING           = 135;
  C_READY                   = 140;
  C_RUNNING                 = 145;
  C_COMPUTED                = 150;
  C_FOR_WU_TRANSMISSION     = 160;
  C_TRANSMITTING_WORKUNIT   = 165;
  C_WORKUNIT_TRANSMITTED    = 170;
  C_FOR_RESULT_TRANSMISSION = 180;
  C_TRANSMITTING_RESULT     = 185;
  C_COMPLETED               = 190;
  C_WORKUNITS_CLEANEDUP     = 800;
  C_ERROR                   = 910;


  // stati when someone else processes a job for us, prefix S_ for server
  S_NEW                     = 200;
  S_FOR_WU_UPLOAD           = 210;
  S_UPLOADING_WORKUNIT      = 215;
  S_FOR_JOB_UPLOAD          = 220;
  S_UPLOADING_JOB           = 225;
  S_FOR_STATUS_RETRIEVAL    = 230;
  S_RETRIEVING_STATUS       = 235;
  S_STATUS_RETRIEVED        = 240;
  S_FOR_WU_RETRIEVAL        = 250;
  S_RETRIEVING_WU           = 255;
  S_FOR_RESULT_RETRIEVAL    = 260;
  S_RETRIEVING_RESULT       = 265;
  S_COMPLETED               = 290;
  S_ERROR                   = 920;



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
  Result := 'Internal error: Unknown Satus in JobQueueStatusToString()';
  case status of
    C_NEW                     : Result := 'C_NEW';
    C_FOR_WU_RETRIEVAL        : Result := 'C_FOR_WU_RETRIEVAL';
    C_RETRIEVING_WORKUNIT     : Result := 'C_RETRIEVING_WORKUNIT';
    C_WORKUNIT_RETRIEVED      : Result := 'C_WORKUNIT_RETRIEVED';
    C_FOR_ACKNOWLEDGEMENT     : Result := 'C_FOR_ACKNOWLEDGEMENT';
    C_ACKNOWLEDGING           : Result := 'C_ACKNOWLEDGING';
    C_READY                   : Result := 'C_READY';
    C_RUNNING                 : Result := 'C_RUNNING';
    C_COMPUTED                : Result := 'C_COMPUTED';
    C_FOR_WU_TRANSMISSION     : Result := 'C_FOR_WU_TRANSMISSION';
    C_TRANSMITTING_WORKUNIT   : Result := 'C_TRANSMITING_WORKUNIT';
    C_WORKUNIT_TRANSMITTED    : Result := 'C_WORKUNIT_TRANSMITTED';
    C_FOR_RESULT_TRANSMISSION : Result := 'C_FOR_RESULT_TRANSMISSION';
    C_TRANSMITTING_RESULT     : Result := 'C_TRANSMITTING_RESULT';
    C_COMPLETED               : Result := 'C_COMPLETED';
    C_WORKUNITS_CLEANEDUP     : Result := 'C_WORKUNITS_CLEANED_UP';
    C_ERROR                   : Result := 'C_ERROR';

    S_NEW                     : Result := 'S_NEW';
    S_FOR_WU_UPLOAD           : Result := 'S_FOR_WU_UPLOAD';
    S_UPLOADING_WORKUNIT      : Result := 'S_UPLOADING_WORKUNIT';
    S_FOR_JOB_UPLOAD          : Result := 'S_FOR_JOB_UPLOAD';
    S_UPLOADING_JOB           : Result := 'S_UPLOADING_JOB';
    S_FOR_STATUS_RETRIEVAL    : Result := 'S_FOR_STATUS_RETRIEVAL';
    S_RETRIEVING_STATUS       : Result := 'S_RETRIEVING_STATUS';
    S_FOR_WU_RETRIEVAL        : Result := 'S_FOR_WU_RETRIEVAL';
    S_RETRIEVING_WU           : Result := 'S_RETRIEVING_WU';
    S_FOR_RESULT_RETRIEVAL    : Result := 'S_FOR_RESULT_RETRIEVAL';
    S_RETRIEVING_RESULT       : Result := 'S_RETRIEVING_RESULT';
    S_COMPLETED               : Result := 'S_COMPLETED';
    S_ERROR                   : Result := 'S_ERROR ';
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
