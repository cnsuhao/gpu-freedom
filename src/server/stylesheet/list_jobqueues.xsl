<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" doctype-public="-//W3C//DTD HTML 4.01//EN"
                doctype-system="http://www.w3.org/TR/html4/strict.dtd" />
 
    <xsl:template match="jobqueues">
        <html>
            <head>
			    <title>GPU Server - List Job Queue</title>
            </head>
            <body>
			    <img src="../images/gpu-inverse.jpg" border="0" />
                <h2>List Job Queues</h2>
                <table border="1">
					<tr>
						<th>id</th>
						<th>jobqueueid</th>
						<th>jobdefinitionid</th>
						<th>workunit job</th>
						<th>workunit result</th>
						<th>requires ack</th>
						<th>create_dt</th>
						<th>transmission_dt</th>
						<th>ack_dt</th>
						<th>nodename of acknowledger</th>
						<th>reception_dt</th>
						<th>job</th>
						<th>nodename of requester</th>
					</tr>
                    <xsl:apply-templates select="jobqueue"/>
                </table>
            </body>
        </html>
    </xsl:template>
 
    <xsl:template match="jobqueue">
        <tr bgcolor="#81F781">
            <td>
                <xsl:value-of select="id"/>			
            </td>
			<td>
                <xsl:value-of select="jobqueueid"/>			
            </td>
			<td>
                <xsl:value-of select="jobdefinitionid"/>			
            </td>
			<td>
                <xsl:value-of select="workunitjob"/>			
            </td>
			<td>
                <xsl:value-of select="workunitresult"/>			
            </td>
			<td>
                <xsl:value-of select="requireack"/>			
            </td>
			<td>
                <xsl:value-of select="create_dt"/>			
            </td>
			<td>
                <xsl:value-of select="transmission_dt"/>			
            </td>	
			<td>
                <xsl:value-of select="ack_dt"/>			
            </td>	
			<td>
                <xsl:value-of select="acknodeid"/>			
            </td>	
			<td>
                <xsl:value-of select="reception_dt"/>			
            </td>		
			<xsl:apply-templates select="jobdefinition"/>
		</tr>
    </xsl:template>
 
    <xsl:template match="jobdefinition">
            <td><xsl:value-of select="job"/></td>
            <td><xsl:value-of select="nodename"/></td>
    </xsl:template>
 
</xsl:stylesheet>