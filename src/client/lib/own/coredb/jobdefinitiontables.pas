unit jobdefinitiontables;
{
   TDbJobTable contains the jobs which need to be executed or are already executed.
   If they need to be executed, there will be a reference in TDbJobQueue.
   If a result was computed, the result will be stored in TDbJobResult with a reference
    to the job.jobtables

   (c) by 2011 HB9TVM and the GPU Team
}
interface

uses sqlite3ds, db, coretables, SysUtils;

type TDbJobDefinitionRow = record
   id         : Longint;
   jobdefinitionid  : String;
   job        : AnsiString;
   jobtype    : String;
   requireack,
   islocal    : Boolean;
   nodeid,
   nodename   : String;
   server_id  : Longint;
   create_dt,
   update_dt  : TDateTime;
end;

type TJobTransmissionDetails = record
     workunitjob : String;
     workunitresult : String;
     nbrequests  : Longint;
     tagwujob,
     tagwuresult : Boolean;
end;

type TDbJobDefinitionTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    function getId(jobid : String) : Longint;
    procedure getRow(var row : TDbJobDefinitionRow);
    procedure insertOrUpdate(var row : TDbJobDefinitionRow);

  private
    procedure createDbTable();
  end;

implementation

constructor TDbJobDefinitionTable.Create(filename : String);
begin
  inherited Create(filename, 'tbjobdefinition', 'id');
  createDbTable();
end;

procedure TDbJobDefinitionTable.createDbTable();
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
      FieldDefs.Add('islocal', ftBoolean);
      FieldDefs.Add('nodeid', ftString);
      FieldDefs.Add('nodename', ftString);
      FieldDefs.Add('server_id', ftInteger);
      FieldDefs.Add('create_dt', ftDateTime);
      FieldDefs.Add('update_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

function TDbJobDefinitionTable.getId(jobid : String) : Longint;
var options : TLocateOptions;
begin
 options := [];
 if dataset_.Locate('jobdefinitionid', jobid, options) then
     Result :=  dataset_.FieldByName('id').AsInteger
    else
      Result := -1;
end;

procedure TDbJobDefinitionTable.getRow(var row : TDbJobDefinitionRow);
var options : TLocateOptions;
    srvid   : Longint;
begin
 options := [];
 if dataset_.Locate('jobdefinitionid', row.jobdefinitionid, options) then
   begin
     row.id         := dataset_.FieldByName('id').AsInteger;
     row.jobdefinitionid  := dataset_.FieldByName('jobdefinitionid').AsString;
     row.job        := dataset_.FieldByName('job').AsString;
     row.jobtype    := dataset_.FieldByName('jobtype').AsString;
     row.requireack := dataset_.FieldByName('requireack').AsBoolean;
     row.islocal    := dataset_.FieldByName('islocal').AsBoolean;
     row.nodeid     := dataset_.FieldByName('nodeid').AsString;
     row.nodename   := dataset_.FieldByName('nodename').AsString;
     row.server_id  := dataset_.FieldByName('server_id').AsInteger;
     row.create_dt  := dataset_.FieldByName('create_dt').AsDateTime;
     row.update_dt  := dataset_.FieldByName('update_dt').AsDateTime;
   end
  else
     row.id := -1;
end;

procedure TDbJobDefinitionTable.insertOrUpdate(var row : TDbJobDefinitionRow);
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
  dataset_.FieldByName('islocal').AsBoolean := row.islocal;
  dataset_.FieldByName('nodeid').AsString := row.nodeid;
  dataset_.FieldByName('nodename').AsString := row.nodename;
  dataset_.FieldByName('server_id').AsInteger := row.server_id;
  dataset_.FieldByName('create_dt').AsDateTime := Now;
  dataset_.FieldByName('update_dt').AsDateTime := Now;

  dataset_.Post;
  dataset_.ApplyUpdates;

  row.id := dataset_.FieldByName('id').AsInteger;
end;


end.
