unit jobresulttables;
{
   TDbJobTable contains the jobs which need to be executed or are already executed.
   If they need to be executed, there will be a reference in TDbJobQueue.
   If a result was computed, the result will be stored in TDbJobResult with a reference
    to the job.jobtables

   (c) by 2011 HB9TVM and the GPU Team
}
interface

uses sqlite3ds, db, coretables, SysUtils;

type TDbJobResultRow = record
   id,
   externalid,
   job_id         : Longint;
   jobresult      : AnsiString;
   workunitresult : String;
   iserroneous    : Boolean;
   errorid        : Longint;
   errormsg,
   errorarg       : String;
   server_id      : Longint;
   create_dt      : TDateTime;
end;

type TDbJobResultTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    procedure getRow(var row : TDbJobResultRow);
    procedure insertOrUpdate(var row : TDbJobResultRow);

  private
    procedure createDbTable();
  end;

implementation


constructor TDbJobResultTable.Create(filename : String);
begin
  inherited Create(filename, 'tbjobresult', 'id');
  createDbTable();
end;

procedure TDbJobResultTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('externalid', ftInteger);
      FieldDefs.Add('job_id', ftInteger);
      FieldDefs.Add('jobresult', ftString);
      FieldDefs.Add('workunitresult', ftString);
      FieldDefs.Add('iserroneous', ftBoolean);
      FieldDefs.Add('errorid', ftInteger);
      FieldDefs.Add('errormsg', ftInteger);
      FieldDefs.Add('errorarg', ftInteger);
      FieldDefs.Add('server_id', ftInteger);
      FieldDefs.Add('create_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

procedure TDbJobResultTable.getRow(var row : TDbJobRow);
var options : TLocateOptions;
begin
 options := [];
 if dataset_.Locate('id', row.id, options) then
   begin
     row.id          := dataset_.FieldByName('id').AsInteger;
     row.externalid  := dataset_.FieldByName('externalid').AsInteger;
     row.job_id      := dataset_.FieldByName('job_id').AsInteger;
     row.jobresult   := dataset_.FieldByName('jobresult').AsString;
     row.workunitresult := dataset_.FieldByName('workunitresult').AsString;
     row.iserroneous := dataset_.FieldByName('iserroneous').AsBoolean;
     row.errorid     := dataset_.FieldByName('errorid').AsInteger;
     row.errormsg    := dataset_.FieldByName('errormsg').AsString;
     row.errorarg    := dataset_.FieldByName('errorarg').AsString;
     row.server_id   := dataset_.FieldByName('server_id').AsInteger;
     row.create_dt   := dataset_.FieldByName('create_dt').AsDateTime;
   end
  else
     row.id := -1;
end;


procedure TDbJobResultTable.insertOrUpdate(var row : TDbJobRow);
var options : TLocateOptions;
begin
  options := [];
  if dataset_.Locate('id', row.id, options) then
      dataset_.Edit
  else
      dataset_.Append;

  dataset_.FieldByName('id').AsInteger := row.id;
  dataset_.FieldByName('externalid').AsInteger := row.externalid;
  dataset_.FieldByName('job_id').AsInteger := row.jobid;
  dataset_.FieldByName('jobresult').AsString := row.job;
  dataset_.FieldByName('workunitresult').AsString := row.workunitresult;
  dataset_.FieldByName('errorid').AsInteger := row.errorid;
  dataset_.FieldByName('errormsg').AsString := row.errormsg;
  dataset_.FieldByName('errorarg').AsString := row.errorarg;
  dataset_.FieldByName('iserroneous').AsBoolean := row.iserroneous;
  dataset_.FieldByName('server_id').AsInteger := row.server_id;
  dataset_.FieldByName('create_dt').AsDateTime := Now;

  dataset_.Post;
  dataset_.ApplyUpdates;
end;

end.
