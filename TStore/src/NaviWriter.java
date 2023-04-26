/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 */

import static org.seaborne.patch.changes.PatchCodes.*;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URISyntaxException;
import org.apache.jena.graph.Node;
import org.apache.jena.graph.Triple;
import org.apache.jena.sparql.core.Quad;
import org.seaborne.patch.text.RDFChangesWriterText;
import org.seaborne.patch.text.TokenWriter;
import org.seaborne.patch.text.TokenWriterText;

/**
 * Write out a changes as a stream of syntax tokens.
 */
public class NaviWriter extends RDFChangesWriterText {

    /** Create a {@code RDFChangesWriter} with standard text output. */
    public static NaviWriter create(OutputStream out) {
        return new NaviWriter(TokenWriterText.create(out));
    }

    protected NaviUpdateOrganizer organizer;

    public NaviWriter(TokenWriter out) {
        super(out);
        organizer = new NaviUpdateOrganizer();
    }

    @Override
    public void close() {
        tok.close();
    }

    @Override
    public void add(Node g, Node s, Node p, Node o) {
        Triple triple = Triple.create(s, p, o);
        organizer.appendTriple(triple, "add");
        output(ADD_DATA, g, s, p, o);
    }

    private void output(String code, Node g, Node s, Node p, Node o) {
        tok.startTuple();
        outputWord(code);
        output(s);
        output(p);
        output(o);
        if ( g != null && ! Quad.isDefaultGraph(g) )
            output(g);
        tok.endTuple();
    }

    private void output(Node node) {
        tok.sendNode(node);
    }

    private void outputWord(String code) {
        tok.sendWord(code);
    }

    @Override
    public void delete(Node g, Node s, Node p, Node o) {
        Triple triple = Triple.create(s, p, o);
        organizer.appendTriple(triple, "delete");
        output(DEL_DATA, g, s, p, o);
    }

    @Override
    public void addPrefix(Node gn, String prefix, String uriStr) {
        tok.startTuple();
        outputWord(ADD_PREFIX);
        tok.sendString(prefix);
        tok.sendString(uriStr);
        if ( gn != null )
            tok.sendNode(gn);
        tok.endTuple();
    }

    @Override
    public void deletePrefix(Node gn, String prefix) {
        tok.startTuple();
        outputWord(DEL_PREFIX);
        tok.sendString(prefix);
        if ( gn != null )
            tok.sendNode(gn);
        tok.endTuple();
    }

    private void oneline(String code) {
        tok.startTuple();
        tok.sendWord(code);
        tok.endTuple();
    }

    @Override
    public void txnBegin() {
        oneline(TXN_BEGIN);
    }

    @Override
    public void txnCommit() {
        try {
            organizer.commit();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        oneline(TXN_COMMIT);
        organizer.clear();
        tok.flush();
    }

    @Override
    public void txnAbort() {
        organizer.clear();
        oneline(TXN_ABORT);
        tok.flush();
    }

    @Override
    public void segment() {
        oneline(SEGMENT);
    }
}
