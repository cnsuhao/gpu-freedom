unit parametertables;
{
   TDbParameterTable contains parameters used by core and frontend.

   (c) by 2010-2013 HB9TVM and the GPU Team
}
interface

uses sqlite3ds, db, coretables, SysUtils, identities;


type TDbParameterRow = record
    id            : Longint;
    paramtype,
    paramname,
    paramvalue    : String;
    create_dt,
    update_dt   : TDateTime;
end;

type TDbParameterTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

    procedure insertorupdate(row : TDbParameterRow);

    function getParameter(paramtype, paramname, ifmissing : String) : String;
    function setParameter(paramtype, paramname, paramvalue : String) : Boolean;

  private
    procedure createDbTable();
  end;

implementation

constructor TDbParameterTable.Create(filename : String);
begin
  inherited Create(filename, 'tbparameter', 'id');
  createDbTable();
end;


procedure TDbParameterTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('paramtype', ftString);
      FieldDefs.Add('paramname', ftString);
      FieldDefs.Add('paramvalue', ftString);
      FieldDefs.Add('create_dt', ftDateTime);
      FieldDefs.Add('update_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

procedure TDbParameterTable.insertorupdate(row : TDbParameterRow);
var options : TLocateOptions;
    updated : Boolean;
begin
   options := [];
   if dataset_.Locate('paramname', row.paramname, options) then
     begin
       dataset_.Edit;
       updated := true;
     end
   else
     begin
       dataset_.Append;
       updated := false;
     end;

  dataset_.FieldByName('paramtype').AsString := row.paramtype;
  dataset_.FieldByName('paramname').AsString := row.paramname;
  dataset_.FieldByName('paramvalue').AsString := row.paramvalue;

  if updated then
    dataset_.FieldByName('update_dt').AsDateTime := Now
  else
    begin
      dataset_.FieldByName('create_dt').AsDateTime := Now;
      //dataset_.FieldByName('update_dt').AsDateTime := nil;
    end;

  dataset_.Post;
  dataset_.ApplyUpdates;
end;


function TDbParameterTable.getParameter(paramtype, paramname, ifmissing : String) : String;
var options : TLocateOptions;
begin
  // TODO: paramtype is currently unused here
  options := [];
  if dataset_.Locate('paramname', paramname, options) then
       begin
         Result :=  dataset_.FieldByName('paramvalue').AsString;
       end
  else
      Result := ifmissing;
end;

function TDbParameterTable.setParameter(paramtype, paramname, paramvalue : String) : Boolean;
var row  : TDbParameterRow;
begin
  row.paramtype  := paramtype;
  row.paramname  := paramname;
  row.paramvalue := paramvalue;

  insertorupdate(row);
end;

end.
