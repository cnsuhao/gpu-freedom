unit jobstatstables;

interface

uses sqlite3ds, db, coretables, SysUtils;

type TDbJobStatsRow = record
   id         : Longint;
   jobdefinitionid  : String;
   job        : AnsiString;
   jobtype    : String;
   requireack : Boolean;
   requests,
   transmitted,
   received,
   acknowledged : Longint;
   server_id    : Longint;
   create_dt    : TDateTime;
end;

type TDbJobStatsTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    procedure insertOrUpdate(var row : TDbJobStatsRow);

  private
    procedure createDbTable();
  end;

implementation

constructor TDbJobStatsTable.Create(filename : String);
begin
  inherited Create(filename, 'tbjobstats', 'id');
  createDbTable();
end;

procedure TDbJobStatsTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('jobdefinitionid', ftString);
      FieldDefs.Add('job', ftString);
      FieldDefs.Add('jobtype', ftString);
      FieldDefs.Add('requireack', ftBoolean);
      FieldDefs.Add('requests', ftInteger);
      FieldDefs.Add('transmitted', ftInteger);
      FieldDefs.Add('received', ftInteger);
      FieldDefs.Add('acknowledged', ftInteger);
      FieldDefs.Add('server_id', ftInteger);
      FieldDefs.Add('create_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;


procedure TDbJobStatsTable.insertOrUpdate(var row : TDbJobStatsRow);
var options : TLocateOptions;
begin
 options := [];
 if dataset_.Locate('jobdefinitionid', row.jobdefinitionid, options) then
     dataset_.Edit
 else
     dataset_.Append;

 dataset_.FieldByName('jobdefinitionid').AsString := row.jobdefinitionid;
 dataset_.FieldByName('job').AsString := row.job;
 dataset_.FieldByName('jobtype').AsString := row.jobtype;
 dataset_.FieldByName('requireack').AsBoolean := row.requireack;
 dataset_.FieldByName('requests').AsInteger := row.requests;
 dataset_.FieldByName('transmitted').AsInteger := row.transmitted;
 dataset_.FieldByName('received').AsInteger := row.received;
 dataset_.FieldByName('acknowledged').AsInteger := row.acknowledged;
 dataset_.FieldByName('server_id').AsInteger := row.server_id;
 dataset_.FieldByName('create_dt').AsDateTime := Now;

 dataset_.Post;
 dataset_.ApplyUpdates;

 row.id := dataset_.FieldByName('id').AsInteger;
end;

end.
