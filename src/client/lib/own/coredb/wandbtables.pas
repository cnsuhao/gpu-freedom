unit wandbtables;
{
   TDbWanParameterTable contains persisted, wide area network parameters, which are
   valid throughout the GPU network. (From an idea of nanobit, implemented in
   GPU I in 0.943 (around 2007)

   (c) by 2007-2013 nanobit (Rene Tegel), HB9TVM and the GPU Team
}
interface

uses sqlite3ds, db, coretables, SysUtils;

type TDbWanParameterRow = record
   id              : Longint;
   wanid,
   paramtype,
   paramname,
   paramvalue     : String;

   nodeid,
   nodename       : String;

   server_id      : Longint;
   create_dt      : TDateTime;
   update_dt      : TDateTime;
end;

type TDbWanParameterTable = class(TDbCoreTable)
  public
    constructor Create(filename : String);

  private
    procedure createDbTable();
  end;

implementation


constructor TDbWanParameterTable.Create(filename : String);
begin
  inherited Create(filename, 'tbwandbparameter', 'id');
  createDbTable();
end;

procedure TDbWanParameterTable.createDbTable();
begin
 with dataset_ do
  begin
    if not TableExists then
    begin
      FieldDefs.Clear;
      FieldDefs.Add('id', ftAutoInc);
      FieldDefs.Add('wanid', ftString);
      FieldDefs.Add('paramtype', ftString);
      FieldDefs.Add('paramname', ftString);
      FieldDefs.Add('paramvalue', ftString);

      FieldDefs.Add('nodeid', ftString);
      FieldDefs.Add('nodename', ftString);

      FieldDefs.Add('server_id', ftInteger);
      FieldDefs.Add('create_dt', ftDateTime);
      FieldDefs.Add('update_dt', ftDateTime);
      CreateTable;
    end; {if not TableExists}
  end; {with}
end;

end.
