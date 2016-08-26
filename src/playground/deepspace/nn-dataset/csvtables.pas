unit csvtables;
{(c) by 2016 HB9TVM an the GPU team. Source code is under GPL}
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, utils, quicksort, Dialogs;

const
     MAX_COLUMNS = 22; // if a table has so many columns, consider
                         // refactoring :-D

type TCSVTable = class(TObject)
   public // we know what we are doing with our datastructure anyway
     filename_,
     header_,
     separator_   : String;

     fields_      : Array[1..MAX_COLUMNS] of AnsiString;
     totalfields_ : Longint;

     totalrows_      : Longint;
     totalcolumns_   : Longint;
     isFloat_,
     singleTest_     : Array[1..MAX_COLUMNS] of Boolean;

     tablemem_       : Array of AnsiString;
     tablememFloat_  : Array[1..MAX_COLUMNS] of Array of Extended;

     constructor Create(filename, separator  : String);
     destructor  Destroy;
     function    readHeader() : AnsiString;
     function    countRows() : Longint;

     function    retrieveNFieldValue(Str : AnsiString; pos : Longint) : AnsiString;
     function    retrieveField(fieldname : AnsiString) : Longint;
     function    retrieveRow(pos : Longint) : AnsiString;
     function    retrieveRowField(startrow,endrow : Longint; field : Ansistring) : AnsiString;
     procedure   loadInMemory;     // fills tablemem_
     procedure   inferFieldsFromData; // fills isFloat_
     procedure   loadInFloatMemory; // fills tablememFloat_

   private
     F : TextFile;
     loadedInMemory_ : Boolean;

end;

implementation

constructor TCSVTable.Create(filename, separator  : String);
var column : AnsiString;
    i      : Longint;
begin
 filename_  := filename;
 separator_ := separator;
 header_    := readHeader();

 loadedInMemory_ := false;

 i:=0;
 column := Trim(ExtractParamLong(header_, separator_));
 while column<>'' do
         begin
           Inc(i);
           fields_[i] := column;
           column := Trim(ExtractParamLong(header_, separator_));
         end;
 totalfields_ := i;
 totalrows_ := countRows();

 loadInMemory();
 inferFieldsFromData();
 loadInFloatMemory();
end;


function TCSVTable.readHeader() : AnsiString;
var str : AnsiString;
begin
  Result := '';
  if Trim(filename_)='' then Exit;
  if not FileExists(filename_) then
        raise Exception.Create('ERROR: filename '+filename_+' does not exist!');

  str := '';
  AssignFile(F, filename_);
  try
    Reset(F);
    ReadLn(F, Str);
  finally
    CloseFile(F);
  end;

  Result := Trim(Str);
end;

function TCSVTable.countRows() : Longint;
var count : Longint;
    str   : AnsiString;
begin
  count := 0;
  AssignFile(F, filename_);
  try
    Reset(F);
    ReadLn(F);  // skip header

    while not EOF(F) do
        begin
          ReadLn(F, Str);
          if Trim(Str)='' then continue;
          Inc(count);
        end;
  finally
    CloseFile(F);
  end;

  Result := count;
end;

procedure TCSVTable.inferFieldsFromData;
var
    str,
    msg : AnsiString;
    j,
    nbcolumns,
    oldnbcolumns : Longint;

    function scanFieldsForNumeric(str : AnsiString) : Longint;
    var column     : AnsiString;
        i          : Longint;
        val        : Extended;
        nbcolumns  : Longint;
        myStr      : Ansistring;
        singleTest : Boolean;
    begin
      myStr := str;
      i := 0;

      column:=extractParamLong(myStr, separator_);
      while (Length(myStr)>0) or (column<>'') do
           begin
             i := i+1;
             if singleTest_[i] then
                begin
                  // test if this column is really numeric
                     try
                        val := StrToFloat(Trim(column));
                        isfloat_[i] := true;

                     except
                           on E : EConvertError do
                             begin
                              isfloat_[i]   := false;
                              singleTest_[i] := false;
                             end;
                     end;
                end;

             // go to next column
             column:=extractParamLong(myStr, separator_);
           end;

       Result := i; // number of columns as result
    end;

begin
  for j:=1 to MAX_COLUMNS do
    begin
      singleTest_[j] := true;
      isFloat_[j] := false;
    end;

  oldnbcolumns := 0;
  for j:=2 to totalrows_ do
      begin
          str := retrieveRow(j);
          if Trim(str)='' then continue;
          nbcolumns:=scanFieldsForNumeric(str);
          if (j>=3) and (oldnbcolumns<>nbcolumns) then
                Raise Exception.Create('The csv file has an unequal number of columns across rows!');
          oldnbcolumns := nbcolumns;
      end;

  totalcolumns_ := nbcolumns;
  {
  // enable this to debug field inference
  msg := '';
  for j:=1 to nbcolumns do
      begin
        msg := msg + IntToStr(j)+' '+fields_[j]+':';
               if isfloat_[j]
                  then msg := msg + 'float '
               else
                       msg := msg + 'text ';
      end;
  ShowMessage(msg);
  }
end;

procedure TCSVTable.loadInFloatMemory();
var i, j : Longint;
    str  : AnsiString;
begin
  for i:=1 to totalcolumns_ do
   begin
      if isFloat_[i] then
         SetLength(tablememFloat_[i], totalrows_); // this goes to 0 to totalrows_-1, 0 contains header
   end;

   // now parse the values into the memory monster:
   for j:=2 to totalrows_ do
       begin
            str := retrieveRow(j);
            if Trim(str)='' then
               begin
                   for i:=1 to totalcolumns_ do
                   if isFloat_[i] then
                      tablememFloat_[i][j-1] := 0;
                   continue;
               end;

            for i:=1 to totalcolumns_ do
                begin
                   if isFloat_[i] then
                      tablememFloat_[i][j-1] := StrToFloat(retrieveNFieldValue(str,i));
                end;
       end;

end;


function TCSVTable.retrieveNFieldValue(Str : AnsiString; pos : Longint) : AnsiString;
var i : Longint;
    value : AnsiString;
begin
  for i:=1 to pos do
     begin
       value := ExtractParamLong(Str, separator_);
     end;
  Result := value;
end;


function TCSVTable.retrieveField(fieldname : AnsiString) : Longint;
var i : Longint;
begin
 Result := -1;
 for i:=1 to totalfields_ do
     if fields_[i]=fieldName then
        begin
           Result:=i;
           Exit;
        end;
end;

function  TCSVTable.retrieveRowField(startrow, endrow : Longint; field : Ansistring) : AnsiString;
var str : AnsiString;
    pos : Longint;
    j   : Longint;
begin
 Result := '';

 pos := retrieveField(field);

 for j:=startrow to endrow do
     begin
        str := retrieveRow(j);
        Result := Result + retrieveNFieldValue(str, pos) + separator_;
     end;
end;


function TCSVTable.retrieveRow(pos : Longint) : AnsiString;
var i : Longint;
    str : AnsiString;
begin
    // much faster random access if table is loaded in memory
    if loadedInMemory_ then
        begin
           Result := tablemem_[pos];
           Exit;
        end;


    AssignFile(F, filename_);

    try
      Reset(F);
      ReadLn(F, Str); // skip header

      i := 0;
      while not EOF(F) do
          begin
            Readln(F, str);
            if Trim(str)='' then continue; // we skip blank lines completely

            if i=pos then
                 begin
                    Result := Str;
                    Exit;  // Exit will call the CloseFile in the finally block!
                 end;
            Inc(i);
          end;

    finally
      CloseFile(F);
    end;


    raise Exception.Create('Internal error: Row '+IntToStr(pos)+' in filename '+filename_+' not found!');
end;

procedure TCSVTable.loadInMemory;
var i : Longint;
    str : AnsiString;
begin
  setLength(tablemem_, totalrows_+1); // we store also the header

  AssignFile(F, filename_);

    try
      Reset(F);
      ReadLn(F, Str); // skip header

      i := 0;
      while not EOF(F) do
          begin
            Readln(F, str);
            if Trim(str)='' then continue; // we skip blank lines completely

            tablemem_[i] := Str;
            Inc(i);
          end;

    finally
      CloseFile(F);
    end;

  loadedInMemory_ := true;
end;


destructor TCSVTable.Destroy;
begin
  if loadedInMemory_ then setLength(tablemem_, 0);
  inherited Destroy;
end;

end.

