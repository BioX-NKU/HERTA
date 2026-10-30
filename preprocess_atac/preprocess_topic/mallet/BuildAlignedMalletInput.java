import cc.mallet.pipe.Noop;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Builds inference instances in the inferencer's TRAINING alphabet, never a new
 * import-file alphabet. Use only trusted MALLET Java serialization inputs.
 *
 * Java 11+ source launch:
 * java -cp "$MALLET_HOME/class:$MALLET_HOME/lib/mallet-deps.jar" \
 *   BuildAlignedMalletInput.java --inferencer model --input docs.txt \
 *   --output new.mallet --report new.json [--oov-policy error|drop]
 *
 * Input is UTF-8: metacell_N<space>X<space>numeric_global_bin ...
 * The delimiter after X is mandatory even for an actually empty document.
 * Numeric strings must be canonical nonnegative decimal integers. They are
 * dictionary KEYS, not MALLET type IDs. No tokenization/training pipe is reused.
 */
public final class BuildAlignedMalletInput {
    private static final int SAMPLE_LIMIT = 16;
    private static final Pattern DOCUMENT = Pattern.compile(
            "^(metacell_[0-9]+)[\\t ]+X[\\t ]+(.*)$");
    private static final Pattern NUMBER = Pattern.compile("0|[1-9][0-9]*");

    private static final class Options {
        Path inferencer, input, output, report;
        String policy = "error";
        Integer expectedTopics;
        Long maxRegionIndex;
    }

    private static final class Model {
        Alphabet alphabet;
        int numTypes, numTopics, topicBits;
        int[][] counts;
        long minRegionIndex = Long.MAX_VALUE, maxRegionIndex = -1;
        String alphabetHash;
        List<Object> mappingSample = new ArrayList<>();
    }

    private static final class Totals {
        long documents, inputTokens, retainedTokens, oovTokens;
        long emptyDocuments, allOovDocuments, missingTokens, untrainedTokens;
        Map<String, Object> json() {
            return map("documents", documents, "input_tokens", inputTokens,
                    "retained_tokens", retainedTokens, "oov_tokens", oovTokens,
                    "empty_documents", emptyDocuments,
                    "all_oov_documents", allOovDocuments,
                    "missing_alphabet_tokens", missingTokens,
                    "without_training_counts_tokens", untrainedTokens);
        }
    }

    public static void main(String[] args) {
        try {
            if (args.length == 1 && args[0].equals("--help")) {
                System.out.println("Usage: BuildAlignedMalletInput.java --inferencer PATH "
                        + "--input PATH --output NEW_PATH --report NEW_PATH "
                        + "[--oov-policy error|drop] [--expected-num-topics N] "
                        + "[--max-region-index N (inclusive)]");
                return;
            }
            boolean success = run(parse(args));
            if (!success) System.exit(2);
        } catch (Exception error) {
            System.err.println("BuildAlignedMalletInput: " + error.getMessage());
            System.exit(2);
        }
    }

    private static Options parse(String[] args) throws Exception {
        Options options = new Options();
        Set<String> seen = new HashSet<>();
        for (int i = 0; i < args.length; i += 2) {
            String flag = args[i];
            if (!seen.add(flag) || i + 1 == args.length)
                throw new IllegalArgumentException("Duplicate option or missing value: " + flag);
            String value = args[i + 1];
            switch (flag) {
                case "--inferencer": options.inferencer = path(value); break;
                case "--input": options.input = path(value); break;
                case "--output": options.output = path(value); break;
                case "--report": options.report = path(value); break;
                case "--oov-policy": options.policy = value; break;
                case "--expected-num-topics":
                    options.expectedTopics = Integer.valueOf(value);
                    if (options.expectedTopics < 1) throw new IllegalArgumentException("Expected topics must be positive");
                    break;
                case "--max-region-index":
                    options.maxRegionIndex = Long.valueOf(value);
                    if (options.maxRegionIndex < 0) throw new IllegalArgumentException("Maximum region index must be nonnegative");
                    break;
                default: throw new IllegalArgumentException("Unknown option: " + flag);
            }
        }
        if (options.inferencer == null || options.input == null || options.output == null || options.report == null)
            throw new IllegalArgumentException("--inferencer, --input, --output and --report are required");
        if (!options.policy.equals("error") && !options.policy.equals("drop"))
            throw new IllegalArgumentException("--oov-policy must be error or drop");
        for (Path source : new Path[] {options.inferencer, options.input}) {
            if (!Files.isRegularFile(source) || !Files.isReadable(source))
                throw new IllegalArgumentException("Unreadable input file: " + source);
        }
        for (Path destination : new Path[] {options.output, options.report}) {
            if (Files.exists(destination, LinkOption.NOFOLLOW_LINKS))
                throw new IllegalArgumentException("Refusing existing destination: " + destination);
            if (!Files.isDirectory(destination.getParent()))
                throw new IllegalArgumentException("Destination parent does not exist: " + destination.getParent());
        }
        // Resolve parent symlinks too: two differently-spelled new paths may alias.
        Path outputIdentity = options.output.getParent().toRealPath().resolve(options.output.getFileName());
        Path reportIdentity = options.report.getParent().toRealPath().resolve(options.report.getFileName());
        if (outputIdentity.equals(reportIdentity)) throw new IllegalArgumentException("Output and report must differ");
        for (Path source : new Path[] {options.input.toRealPath(), options.inferencer.toRealPath()}) {
            if (source.equals(outputIdentity) || source.equals(reportIdentity))
                throw new IllegalArgumentException("A destination aliases an input");
        }
        return options;
    }

    private static boolean run(Options options) throws Exception {
        String beforeHash = hashFile(options.inferencer);
        String afterHash = null;
        Model model = null;
        Totals totals = new Totals();
        Path reportTemp = Files.createTempFile(options.report.getParent(), ".aligned-report-", ".tmp");
        Path outputTemp = null;
        boolean outputPublished = false;
        String error = null;
        InstanceList instances = null;
        try {
            try (BufferedWriter report = Files.newBufferedWriter(reportTemp, StandardCharsets.UTF_8)) {
                report.write("{\"schema_version\":1,\"documents\":[");
                try {
                    // Read exactly once. We never write or save this inferencer.
                    TopicInferencer inferencer = TopicInferencer.read(options.inferencer.toFile());
                    model = inspectModel(inferencer, options);
                    model.alphabet.stopGrowth(); // Only the deserialized in-memory object.
                    instances = new InstanceList(new Noop(model.alphabet, null));
                    readDocuments(options, model, instances, totals, report);
                    if (totals.documents == 0) throw new IllegalArgumentException("Input contains no documents");
                    if (totals.allOovDocuments != 0)
                        throw new IllegalArgumentException("Nonempty documents became all-OOV; refusing prior-only inference under any OOV policy");
                    if (totals.oovTokens != 0 && options.policy.equals("error"))
                        throw new IllegalArgumentException("OOV tokens found; no output created (explicit --oov-policy drop permits partial OOV documents)");
                    if (instances.getDataAlphabet() != model.alphabet)
                        throw new IllegalStateException("Output dictionary identity changed");
                    outputTemp = Files.createTempFile(options.output.getParent(), ".aligned-input-", ".tmp");
                    // InstanceList.save can swallow I/O errors in some MALLET versions.
                    try (ObjectOutputStream stream = new ObjectOutputStream(Files.newOutputStream(outputTemp))) {
                        stream.writeObject(instances);
                    }
                } catch (Exception failure) {
                    error = failure.getClass().getSimpleName() + ": " + failure.getMessage();
                }
                afterHash = hashFile(options.inferencer);
                if (!beforeHash.equals(afterHash)) error = "Inferencer changed during input construction";
                report.write("],\"status\":");
                writeJson(report, error == null ? "ok" : "error");
                report.write(",\"error\":"); writeJson(report, error);
                report.write(",\"oov_policy\":"); writeJson(report, options.policy);
                report.write(",\"inferencer_unchanged\":" + beforeHash.equals(afterHash));
                report.write(",\"input\":"); writeJson(report, options.input.toString());
                report.write(",\"output\":"); writeJson(report, options.output.toString());
                report.write(",\"model\":"); writeJson(report, model == null ? null : modelJson(model));
                report.write(",\"totals\":"); writeJson(report, totals.json());
                report.write("}\n");
            }
            if (error == null) {
                publishNew(outputTemp, options.output);
                outputTemp = null;
                outputPublished = true;
            }
            try {
                publishNew(reportTemp, options.report);
                reportTemp = null;
            } catch (Exception failure) {
                if (outputPublished) Files.delete(options.output);
                throw failure;
            }
            Map<String, Object> summary = map("status", error == null ? "ok" : "error",
                    "error", error, "report", options.report.toString(),
                    "model", model == null ? null : modelJson(model), "totals", totals.json());
            BufferedWriter stdout = new BufferedWriter(new java.io.OutputStreamWriter(System.out, StandardCharsets.UTF_8));
            writeJson(stdout, summary); stdout.write("\n"); stdout.flush();
            if (error != null) System.err.println("BuildAlignedMalletInput: " + error);
            return error == null;
        } finally {
            if (reportTemp != null) Files.deleteIfExists(reportTemp);
            if (outputTemp != null) Files.deleteIfExists(outputTemp);
        }
    }

    private static Model inspectModel(TopicInferencer inferencer, Options options) throws Exception {
        Model model = new Model();
        model.alphabet = (Alphabet) field(inferencer, "alphabet");
        model.numTypes = (Integer) field(inferencer, "numTypes");
        model.numTopics = (Integer) field(inferencer, "numTopics");
        model.topicBits = (Integer) field(inferencer, "topicBits");
        model.counts = (int[][]) field(inferencer, "typeTopicCounts");
        if (model.alphabet == null || model.alphabet.size() == 0 || model.numTopics < 1
                || model.numTypes < 0 || model.counts == null || model.counts.length != model.numTypes
                || model.topicBits < 0 || model.topicBits > 30)
            throw new IllegalArgumentException("Invalid inferencer alphabet or topic/type dimensions");
        if (options.expectedTopics != null && model.numTopics != options.expectedTopics)
            throw new IllegalArgumentException("Expected " + options.expectedTopics + " topics, model has " + model.numTopics);
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        for (int id = 0; id < model.alphabet.size(); id++) {
            Object entry = model.alphabet.lookupObject(id);
            if (!(entry instanceof String)) throw new IllegalArgumentException("Training alphabet key is not a String at ID " + id);
            String token = (String) entry;
            long region = regionIndex(token, options.maxRegionIndex);
            model.minRegionIndex = Math.min(model.minRegionIndex, region);
            model.maxRegionIndex = Math.max(model.maxRegionIndex, region);
            // Canonical numeric strings eliminate normalized aliases such as 01/1.
            // Incremental, order-sensitive digest; no whole-alphabet string copy.
            byte[] bytes = token.getBytes(StandardCharsets.UTF_8);
            updateInt(digest, id); updateInt(digest, bytes.length); digest.update(bytes);
            if (id < SAMPLE_LIMIT) model.mappingSample.add(map("token", token, "type_id", id));
        }
        model.alphabetHash = hex(digest.digest());
        return model;
    }

    private static void readDocuments(Options options, Model model, InstanceList instances,
                                      Totals totals, BufferedWriter report) throws Exception {
        Set<String> names = new HashSet<>();
        try (BufferedReader reader = Files.newBufferedReader(options.input, StandardCharsets.UTF_8)) {
            String line;
            long lineNumber = 0;
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                Matcher match = DOCUMENT.matcher(line);
                if (!match.matches()) throw new IllegalArgumentException("Malformed HERTA document at line " + lineNumber
                        + "; expected metacell_N X plus a delimited (possibly empty) token field");
                String name = match.group(1);
                if (!names.add(name)) throw new IllegalArgumentException("Duplicate document name at line " + lineNumber + ": " + name);
                FeatureSequence sequence = new FeatureSequence(model.alphabet);
                StringTokenizer tokens = new StringTokenizer(match.group(2));
                long inputCount = 0, oovCount = 0;
                List<Object> ids = new ArrayList<>();
                List<Object> oov = new ArrayList<>();
                while (tokens.hasMoreTokens()) {
                    String token = tokens.nextToken();
                    try { regionIndex(token, options.maxRegionIndex); }
                    catch (IllegalArgumentException failure) {
                        throw new IllegalArgumentException("Line " + lineNumber + ": " + failure.getMessage());
                    }
                    inputCount++;
                    int id = model.alphabet.lookupIndex(token, false); // NEVER add.
                    String reason = id < 0 ? "missing_alphabet_token" : hasTrainingCounts(model, id) ? null : "without_training_counts";
                    if (reason != null) {
                        oovCount++;
                        if (id < 0) totals.missingTokens++; else totals.untrainedTokens++;
                        if (oov.size() < SAMPLE_LIMIT) oov.add(map("token", token, "type_id", id, "reason", reason));
                    } else {
                        sequence.add(id); // Exact trained type ID, preserving token order/multiplicity.
                        if (ids.size() < SAMPLE_LIMIT) ids.add(id);
                    }
                }
                boolean empty = inputCount == 0;
                boolean allOov = !empty && sequence.size() == 0;
                if (totals.documents != 0) report.write(",");
                writeJson(report, map("name", name, "line", lineNumber, "input_tokens", inputCount,
                        "retained_tokens", sequence.size(), "oov_tokens", oovCount,
                        "empty_document", empty, "all_oov", allOov,
                        "type_ids_sample", ids, "oov_sample", oov));
                totals.documents++;
                totals.inputTokens += inputCount; totals.retainedTokens += sequence.size(); totals.oovTokens += oovCount;
                if (empty) totals.emptyDocuments++;
                if (allOov) totals.allOovDocuments++;
                instances.add(new Instance(sequence, null, name, null));
            }
        }
    }

    private static boolean hasTrainingCounts(Model model, int id) {
        if (id < 0 || id >= model.numTypes || id >= model.counts.length || model.counts[id] == null) return false;
        // A present-but-zero packed row is not evidence of training counts.
        for (int packed : model.counts[id]) if ((packed >>> model.topicBits) > 0) return true;
        return false;
    }

    private static long regionIndex(String token, Long maximum) {
        if (!NUMBER.matcher(token).matches()) throw new IllegalArgumentException("Noncanonical numeric global-bin token: " + token);
        final long value;
        try { value = Long.parseLong(token); }
        catch (NumberFormatException failure) { throw new IllegalArgumentException("Global-bin token exceeds signed 64-bit range: " + token); }
        if (maximum != null && value > maximum) throw new IllegalArgumentException("Global-bin token " + token + " exceeds --max-region-index " + maximum);
        return value;
    }

    private static Map<String, Object> modelJson(Model model) {
        return map("num_topics", model.numTopics, "num_types", model.numTypes,
                "alphabet_size", model.alphabet.size(), "alphabet_tokens_numeric", true,
                "alphabet_min_region_index", model.minRegionIndex,
                "alphabet_max_region_index", model.maxRegionIndex,
                "mapping_sample", model.mappingSample);
    }

    private static Object field(Object value, String name) throws Exception {
        Field field = TopicInferencer.class.getDeclaredField(name);
        field.setAccessible(true);
        return field.get(value);
    }
    private static void publishNew(Path temporary, Path destination) throws Exception {
        boolean created = false;
        try {
            // CREATE_NEW checks existence and creates atomically, including against
            // symlinks. A plain move can race an intervening destination creation.
            try (OutputStream stream = Files.newOutputStream(destination,
                    StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
                created = true;
                Files.copy(temporary, stream);
            }
            Files.delete(temporary);
        } catch (Exception failure) {
            if (created) Files.deleteIfExists(destination);
            throw failure;
        }
    }
    private static Path path(String text) { return Paths.get(text).toAbsolutePath().normalize(); }
    private static void updateInt(MessageDigest digest, int number) {
        for (int shift = 24; shift >= 0; shift -= 8) digest.update((byte) (number >>> shift));
    }
    private static String hashFile(Path file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] buffer = new byte[1024 * 1024];
        try (InputStream stream = Files.newInputStream(file)) {
            int count;
            while ((count = stream.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        return hex(digest.digest());
    }
    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) result.append(String.format("%02x", value & 255));
        return result.toString();
    }
    private static Map<String, Object> map(Object... pairs) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) result.put((String) pairs[i], pairs[i + 1]);
        return result;
    }
    private static void writeJson(java.io.Writer out, Object value) throws Exception {
        if (value == null) { out.write("null"); return; }
        if (value instanceof Number || value instanceof Boolean) { out.write(value.toString()); return; }
        if (value instanceof Map) {
            out.write("{"); boolean first = true;
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
                if (!first) out.write(","); first = false;
                writeJson(out, entry.getKey().toString()); out.write(":"); writeJson(out, entry.getValue());
            }
            out.write("}"); return;
        }
        if (value instanceof Iterable) {
            out.write("["); boolean first = true;
            for (Object element : (Iterable<?>) value) {
                if (!first) out.write(","); first = false; writeJson(out, element);
            }
            out.write("]"); return;
        }
        out.write('"');
        for (char character : value.toString().toCharArray()) {
            switch (character) {
                case '"': out.write("\\\""); break;
                case '\\': out.write("\\\\"); break;
                case '\n': out.write("\\n"); break;
                case '\r': out.write("\\r"); break;
                case '\t': out.write("\\t"); break;
                default:
                    if (character < 32) out.write(String.format("\\u%04x", (int) character));
                    else out.write(character);
            }
        }
        out.write('"');
    }
}
