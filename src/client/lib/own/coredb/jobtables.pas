unit jobtables;
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
  JS_NEW             = 10;
  JS_TRANSMITTED     = 20;
  JS_RECEIVED        = 30;
  JS_RUNNING         = 40;
  JS_ONHOLD          = 50;
  JS_COMPLETED       = 90;


type TDbJobRow = record
   id         : Longint;
   externalid : String;
   jobid      : String;
   job        : AnsiString;
   status     : Longint;
   workunitincoming,
   workunitoutgoing : String;
   requests,
   delivered,
   results    : Longint;
   islocal    : Boolean;
   server_id  : Longint;
   create_dt  : TDateTime;
end;

type TDbJobTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    procedure getRow(var row : TDbJobRow);
    procedure insertOrUpdate(var row : TDbJobRow);

  private
    procedure createDbTable();
  end;

implementation

constructor TDbJobTable.Create(filename : String);
begin
  inherited Create(filename, 'tbjob', 'id');
  createDbTable();
end;

procedure TDbJobTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('externalid', ftString);
      FieldDefs.Add('jobid', ftString);
      FieldDefs.Add('job', ftString);
      FieldDefs.Add('status', ftInteger);
      FieldDefs.Add('workunitincoming', ftString);
      FieldDefs.Add('workunitoutgoing', ftString);
      FieldDefs.Add('requests', ftInteger);
      FieldDefs.Add('delivered', ftInteger);
      FieldDefs.Add('results', ftInteger);
      FieldDefs.Add('islocal', ftBoolean);
      FieldDefs.Add('server_id', ftInteger);
      FieldDefs.Add('create_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

procedure TDbJobTable.getRow(var row : TDbJobRow);
var options : TLocateOptions;
begin
 options := [];
 if dataset_.Locate('externalid', row.externalid, options) then
   begin
     row.id         := dataset_.FieldByName('id').AsInteger;
     row.externalid := dataset_.FieldByName('externalid').AsString;
     row.jobid      := dataset_.FieldByName('jobid').AsString;
     row.job        := dataset_.FieldByName('job').AsString;
     row.status     := dataset_.FieldByName('status').AsInteger;
     row.requests   := dataset_.FieldByName('requests').AsInteger;
     row.delivered  := dataset_.FieldByName('delivered').AsInteger;
     row.results    := dataset_.FieldByName('results').AsInteger;
     row.workunitincoming := dataset_.FieldByName('workunitincoming').AsString;
     row.workunitoutgoing := dataset_.FieldByName('workunitoutgoing').AsString;
     row.islocal          := dataset_.FieldByName('islocal').AsBoolean;
     row.server_id  := dataset_.FieldByName('server_id').AsInteger;
     row.create_dt  := dataset_.FieldByName('create_dt').AsDateTime;
   end
  else
     row.id := -1;
end;

procedure TDbJobTable.insertOrUpdate(var row : TDbJobRow);
var options : TLocateOptions;
begin
  options := [];
  if dataset_.Locate('externalid', row.externalid, options) then
      dataset_.Edit
  else
      dataset_.Append;

  dataset_.FieldByName('externalid').AsString := row.externalid;
  dataset_.FieldByName('jobid').AsString := row.jobid;
  dataset_.FieldByName('job').AsString := row.job;
  dataset_.FieldByName('status').AsInteger := row.status;
  dataset_.FieldByName('requests').AsInteger := row.requests;
  dataset_.FieldByName('delivered').AsInteger := row.delivered;
  dataset_.FieldByName('results').AsInteger := row.results;
  dataset_.FieldByName('workunitincoming').AsString := row.workunitincoming;
  dataset_.FieldByName('workunitoutgoing').AsString := row.workunitoutgoing;
  dataset_.FieldByName('islocal').AsBoolean := row.islocal;
  dataset_.FieldByName('server_id').AsInteger := row.server_id;
  dataset_.FieldByName('create_dt').AsDateTime := Now;

  dataset_.Post;
  dataset_.ApplyUpdates;

  row.id := dataset_.FieldByName('id').AsInteger;
end;


end.
