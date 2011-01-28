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

type TDbJobQueueRow = record
   id,
   job_id : Longint;
   external_id : Longint;
   create_dt : TDateTime;
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
      FieldDefs.Add('job_id', ftInteger);
      FieldDefs.Add('external_id', ftInteger);
      FieldDefs.Add('create_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

procedure TDbJobQueueTable.insert(var row : TDbJobQueueRow);
begin
  dataset_.Append;
  dataset_.FieldByName('job_id').AsInteger := row.job_id;
  dataset_.FieldByName('external_id').AsInteger := row.external_id;
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
