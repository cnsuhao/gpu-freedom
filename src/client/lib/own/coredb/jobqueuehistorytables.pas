unit jobqueuehistorytables;
{
   TDbJobQueueHistoryTable contains the history of status changes for a jobqueue

   (c) by 2013 HB9TVM and the GPU Team
}
interface

uses sqlite3ds, db, coretables, jobqueuetables, SysUtils;


type TDbJobQueueHistoryRow = record
   id              : Longint;
   jobqueueid      : String;
   status          : Longint;
   statusdesc,
   message         : String;
   create_dt       : TDateTime;
end;

type TDbJobQueueHistoryTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    procedure insert(var row : TDbJobQueueHistoryRow);
  private
    procedure createDbTable();
  end;

implementation

constructor TDbJobQueueHistoryTable.Create(filename : String);
begin
  inherited Create(filename, 'tbjobqueuehistory', 'id');
  createDbTable();
end;

procedure TDbJobQueueHistoryTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('jobqueueid', ftString);
      FieldDefs.Add('status', ftInteger);
      FieldDefs.Add('statusdesc', ftString);
      FieldDefs.Add('message', ftString);
      FieldDefs.Add('create_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

procedure TDbJobQueueHistoryTable.insert(var row : TDbJobQueueHistoryRow);
begin
  dataset_.Append;
  dataset_.FieldByName('jobqueueid').AsString := row.jobqueueid;
  dataset_.FieldByName('status').AsInteger := row.status;
  dataset_.FieldByName('statusdesc').AsString := JobQueueStatusToString(row.status);
  dataset_.FieldByName('message').AsString := row.message;
  dataset_.FieldByName('create_dt').AsDateTime := Now;
  dataset_.Post;
  dataset_.ApplyUpdates;
  row.id := dataset_.FieldByName('id').AsInteger;
end;

end.
