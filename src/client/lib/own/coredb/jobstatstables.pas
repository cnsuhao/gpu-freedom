unit jobstatstables;

interface

uses sqlite3ds, db, coretables, SysUtils;

type TDbJobStatsRow = record
   id         : Longint;
   jobdefinitionid  : String;
   job        : AnsiString;
   jobtype    : String;
   requireack : Boolean;
   transmitted,
   received,
   acknowledged : Longint;
   server_id    : Longint;
   create_dt    : TDateTime;
end;

type TDbJobStatsTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

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
      FieldDefs.Add('transmitted', ftInteger);
      FieldDefs.Add('received', ftInteger);
      FieldDefs.Add('acknowledged', ftInteger);
      FieldDefs.Add('server_id', ftInteger);
      FieldDefs.Add('create_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

end.
