package org.dongguk.common.business;

import java.io.File;
import java.util.Comparator;

/**
 * 파일을 이름순으로 정렬하기 위한 Comparator
 */
public class ProblemFileComparator implements Comparator<File> {
    private static final AlphaNumericStringComparator ALPHA_NUMERIC_STRING_COMPARATOR = new AlphaNumericStringComparator();
    private static final Comparator<File> COMPARATOR = Comparator.comparing(File::getParent, ALPHA_NUMERIC_STRING_COMPARATOR)
            .thenComparing(File::isDirectory)
            .thenComparing(f -> !f.getName().toLowerCase().startsWith("demo"))
            .thenComparing(f -> f.getName().toLowerCase(), ALPHA_NUMERIC_STRING_COMPARATOR)
            .thenComparing(File::getName);

    @Override
    public int compare(File a, File b) {
        return COMPARATOR.compare(a, b);
    }
}
